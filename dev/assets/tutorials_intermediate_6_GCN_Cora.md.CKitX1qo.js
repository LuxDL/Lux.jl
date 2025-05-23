import{_ as s,c as n,o as e,al as p}from"./chunks/framework.Dgw_Mll3.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-05-23 22:12:50.789408: I external/xla/xla/service/service.cc:152] XLA service 0x15c0c080 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-05-23 22:12:50.789590: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1748038370.790815  647952 se_gpu_pjrt_client.cc:1026] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1748038370.790946  647952 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748038370.790994  647952 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748038370.804420  647952 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-2/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-2/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-05-23 22:14:00.046160: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 104 bytes spill stores, 104 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:00.481625: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:00.681331: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:01.062809: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 1212 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:01.376030: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 284 bytes spill stores, 284 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:01.397969: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 272 bytes spill stores, 272 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:01.606214: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:01.968580: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 980 bytes spill stores, 976 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:02.304319: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:02.523055: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 996 bytes spill stores, 968 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:02.593990: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 292 bytes spill stores, 292 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:02.651181: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:02.882793: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 604 bytes spill stores, 608 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:03.294896: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_17&#39;, 48 bytes spill stores, 48 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:04.023969: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:04.551389: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:04.652714: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:04.857931: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:04.953801: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_20&#39;, 360 bytes spill stores, 356 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:14:05.195689: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1748038445.419211  647952 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748038445.419275  647952 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748038445.419278  647952 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748038445.419281  647952 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748038445.419284  647952 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748038445.419287  647952 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748038445.419290  647952 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748038445.419293  647952 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748038445.419296  647952 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748038445.419299  647952 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-05-23 22:14:05.419311: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.422187  647952 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748038445.422230  647952 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748038445.422233  647952 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748038445.422236  647952 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748038445.422239  647952 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748038445.422242  647952 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748038445.422245  647952 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748038445.422247  647952 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748038445.422250  647952 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748038445.422253  647952 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-05-23 22:14:05.422263: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.424987  647952 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748038445.425025  647952 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748038445.425028  647952 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748038445.425031  647952 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748038445.425034  647952 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748038445.425037  647952 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748038445.425039  647952 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748038445.425042  647952 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748038445.425045  647952 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748038445.425048  647952 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-05-23 22:14:05.425058: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.427640  647952 buffer_comparator.cc:145] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1748038445.427681  647952 buffer_comparator.cc:145] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1748038445.427684  647952 buffer_comparator.cc:145] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1748038445.427687  647952 buffer_comparator.cc:145] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1748038445.427690  647952 buffer_comparator.cc:145] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748038445.427693  647952 buffer_comparator.cc:145] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1748038445.427696  647952 buffer_comparator.cc:145] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1748038445.427699  647952 buffer_comparator.cc:145] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1748038445.427702  647952 buffer_comparator.cc:145] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1748038445.427705  647952 buffer_comparator.cc:145] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-05-23 22:14:05.427712: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.430311  647952 buffer_comparator.cc:145] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1748038445.430345  647952 buffer_comparator.cc:145] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1748038445.430349  647952 buffer_comparator.cc:145] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1748038445.430351  647952 buffer_comparator.cc:145] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1748038445.430354  647952 buffer_comparator.cc:145] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1748038445.430357  647952 buffer_comparator.cc:145] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1748038445.430360  647952 buffer_comparator.cc:145] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1748038445.430363  647952 buffer_comparator.cc:145] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1748038445.430366  647952 buffer_comparator.cc:145] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1748038445.430369  647952 buffer_comparator.cc:145] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-05-23 22:14:05.430376: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.433014  647952 buffer_comparator.cc:145] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1748038445.433044  647952 buffer_comparator.cc:145] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1748038445.433047  647952 buffer_comparator.cc:145] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1748038445.433050  647952 buffer_comparator.cc:145] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1748038445.433053  647952 buffer_comparator.cc:145] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1748038445.433056  647952 buffer_comparator.cc:145] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1748038445.433059  647952 buffer_comparator.cc:145] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1748038445.433062  647952 buffer_comparator.cc:145] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1748038445.433065  647952 buffer_comparator.cc:145] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1748038445.433068  647952 buffer_comparator.cc:145] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-05-23 22:14:05.433075: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.435719  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.435760  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.435764  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.435767  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.435770  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.435773  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.435776  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.435779  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.435782  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.435785  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.435793: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.438597  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.438685  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.438689  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.438692  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.438695  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.438697  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.438700  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.438703  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.438706  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.438709  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.438720: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.441807  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.441880  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.441883  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.441886  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.441889  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.441892  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.441895  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.441898  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.441900  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.441903  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.441915: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.444916  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.444970  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.444973  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.444977  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.444979  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.444982  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.444987  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.444990  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.444993  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.444996  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.445007: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.447703  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.447746  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.447749  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.447752  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.447755  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.447758  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.447760  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.447763  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.447766  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.447769  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.447778: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.450393  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.450444  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.450447  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.450450  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.450453  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.450456  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.450459  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.450462  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.450464  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.450467  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.450476: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.453084  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.453125  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.453128  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.453131  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.453134  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.453137  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.453140  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.453143  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.453146  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.453149  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.453159: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.455805  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.455848  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.455851  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.455854  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.455857  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.455860  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.455862  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.455865  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.455868  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.455871  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.455879: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.458542  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.458593  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.458596  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.458599  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.458602  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.458605  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.458608  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.458611  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.458614  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.458617  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.458625: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.461274  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.461316  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.461320  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.461323  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.461326  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.461328  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.461331  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.461334  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.461337  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.461340  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.461348: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.464004  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.464043  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.464048  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.464051  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.464054  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.464057  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.464059  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.464062  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.464065  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.464068  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.464076: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.466707  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.466756  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.466759  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.466762  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.466765  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.466768  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.466770  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.466773  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.466776  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.466779  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.466787: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.469413  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.469453  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.469457  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.469460  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.469463  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.469465  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.469468  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.469471  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.469474  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.469477  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.469484: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.472083  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.472122  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.472125  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.472128  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.472131  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.472134  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.472138  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.472141  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.472144  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.472147  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.472155: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.474766  647952 buffer_comparator.cc:145] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1748038445.474803  647952 buffer_comparator.cc:145] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1748038445.474806  647952 buffer_comparator.cc:145] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1748038445.474809  647952 buffer_comparator.cc:145] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1748038445.474812  647952 buffer_comparator.cc:145] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1748038445.474815  647952 buffer_comparator.cc:145] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1748038445.474818  647952 buffer_comparator.cc:145] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1748038445.474820  647952 buffer_comparator.cc:145] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1748038445.474823  647952 buffer_comparator.cc:145] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1748038445.474826  647952 buffer_comparator.cc:145] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-05-23 22:14:05.474834: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.477472  647952 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1748038445.477506  647952 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1748038445.477509  647952 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1748038445.477512  647952 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1748038445.477515  647952 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1748038445.477518  647952 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1748038445.477521  647952 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1748038445.477524  647952 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1748038445.477526  647952 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1748038445.477529  647952 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-05-23 22:14:05.477536: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.480083  647952 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1748038445.480112  647952 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1748038445.480116  647952 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1748038445.480118  647952 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1748038445.480121  647952 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1748038445.480124  647952 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1748038445.480127  647952 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1748038445.480130  647952 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1748038445.480133  647952 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1748038445.480136  647952 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-05-23 22:14:05.480142: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.482670  647952 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1748038445.482698  647952 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1748038445.482701  647952 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1748038445.482704  647952 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1748038445.482707  647952 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1748038445.482710  647952 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1748038445.482713  647952 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1748038445.482715  647952 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1748038445.482718  647952 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1748038445.482721  647952 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-05-23 22:14:05.482727: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.485278  647952 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1748038445.485311  647952 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1748038445.485315  647952 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1748038445.485318  647952 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1748038445.485320  647952 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1748038445.485323  647952 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1748038445.485326  647952 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1748038445.485329  647952 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1748038445.485332  647952 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1748038445.485335  647952 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-05-23 22:14:05.485342: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.488000  647952 buffer_comparator.cc:145] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1748038445.488043  647952 buffer_comparator.cc:145] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1748038445.488046  647952 buffer_comparator.cc:145] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1748038445.488049  647952 buffer_comparator.cc:145] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1748038445.488052  647952 buffer_comparator.cc:145] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1748038445.488055  647952 buffer_comparator.cc:145] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1748038445.488058  647952 buffer_comparator.cc:145] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1748038445.488061  647952 buffer_comparator.cc:145] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1748038445.488064  647952 buffer_comparator.cc:145] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1748038445.488067  647952 buffer_comparator.cc:145] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-05-23 22:14:05.488074: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.490704  647952 buffer_comparator.cc:145] Difference at 0: 21.7575, expected 19.3855</span></span>
<span class="line"><span>E0000 00:00:1748038445.490731  647952 buffer_comparator.cc:145] Difference at 3: 14.319, expected 17.5973</span></span>
<span class="line"><span>E0000 00:00:1748038445.490735  647952 buffer_comparator.cc:145] Difference at 9: 14.8402, expected 16.6531</span></span>
<span class="line"><span>E0000 00:00:1748038445.490738  647952 buffer_comparator.cc:145] Difference at 20: 13.7726, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1748038445.490741  647952 buffer_comparator.cc:145] Difference at 26: 15.2226, expected 17.2903</span></span>
<span class="line"><span>E0000 00:00:1748038445.490744  647952 buffer_comparator.cc:145] Difference at 27: 18.7304, expected 16.5311</span></span>
<span class="line"><span>E0000 00:00:1748038445.490747  647952 buffer_comparator.cc:145] Difference at 31: 14.8392, expected 16.8073</span></span>
<span class="line"><span>E0000 00:00:1748038445.490750  647952 buffer_comparator.cc:145] Difference at 33: 12.4405, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1748038445.490753  647952 buffer_comparator.cc:145] Difference at 39: 19.1851, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1748038445.490756  647952 buffer_comparator.cc:145] Difference at 41: 17.4688, expected 20.3484</span></span>
<span class="line"><span>2025-05-23 22:14:05.490762: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.493417  647952 buffer_comparator.cc:145] Difference at 128: 0.278558, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1748038445.493438  647952 buffer_comparator.cc:145] Difference at 129: 0.766922, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1748038445.493442  647952 buffer_comparator.cc:145] Difference at 130: 0.824242, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1748038445.493445  647952 buffer_comparator.cc:145] Difference at 131: 0.934478, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1748038445.493448  647952 buffer_comparator.cc:145] Difference at 132: 0.683298, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1748038445.493451  647952 buffer_comparator.cc:145] Difference at 133: 0.107889, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1748038445.493454  647952 buffer_comparator.cc:145] Difference at 134: 0.716831, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1748038445.493457  647952 buffer_comparator.cc:145] Difference at 135: 0.182228, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1748038445.493459  647952 buffer_comparator.cc:145] Difference at 136: 0.780881, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1748038445.493462  647952 buffer_comparator.cc:145] Difference at 137: 0.0990953, expected 18.5916</span></span>
<span class="line"><span>2025-05-23 22:14:05.493467: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.501534  647952 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1748038445.501580  647952 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1748038445.501583  647952 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1748038445.501586  647952 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1748038445.501589  647952 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1748038445.501592  647952 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1748038445.501595  647952 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1748038445.501598  647952 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1748038445.501600  647952 buffer_comparator.cc:145] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1748038445.501603  647952 buffer_comparator.cc:145] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-05-23 22:14:05.501613: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.508178  647952 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1748038445.508219  647952 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1748038445.508222  647952 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1748038445.508225  647952 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1748038445.508228  647952 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1748038445.508233  647952 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1748038445.508236  647952 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1748038445.508239  647952 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1748038445.508241  647952 buffer_comparator.cc:145] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1748038445.508244  647952 buffer_comparator.cc:145] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-05-23 22:14:05.508253: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.514766  647952 buffer_comparator.cc:145] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1748038445.514796  647952 buffer_comparator.cc:145] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1748038445.514799  647952 buffer_comparator.cc:145] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1748038445.514802  647952 buffer_comparator.cc:145] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1748038445.514805  647952 buffer_comparator.cc:145] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1748038445.514808  647952 buffer_comparator.cc:145] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1748038445.514811  647952 buffer_comparator.cc:145] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1748038445.514814  647952 buffer_comparator.cc:145] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1748038445.514817  647952 buffer_comparator.cc:145] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1748038445.514819  647952 buffer_comparator.cc:145] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-05-23 22:14:05.514826: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.519761  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.519795  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.519798  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.519801  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.519804  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.519807  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.519810  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.519813  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.519816  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.519819  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.519826: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.524786  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.524811  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.524815  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.524817  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.524820  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.524823  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.524826  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.524829  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.524832  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.524836  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.524843: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.529609  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.529639  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.529642  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.529645  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.529648  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.529651  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.529653  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.529656  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.529659  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.529662  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.529669: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.534420  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.534446  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.534449  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.534452  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.534455  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.534458  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.534461  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.534464  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.534467  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.534469  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.534477: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.539632  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.539664  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.539667  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.539670  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.539672  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.539675  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.539678  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.539681  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.539684  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.539687  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.539694: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.544659  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.544695  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.544699  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.544702  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.544704  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.544707  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.544710  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.544713  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.544716  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.544719  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.544726: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.550163  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.550191  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.550194  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.550197  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.550200  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.550202  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.550205  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.550208  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.550211  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.550214  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.550221: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.555477  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.555499  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.555503  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.555505  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.555508  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.555511  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.555514  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.555517  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.555520  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.555523  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.555528: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.561231  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.561256  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.561259  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.561262  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.561265  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.561269  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.561272  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.561274  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.561277  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.561280  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.561287: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.565909  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.565927  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.565930  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.565933  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.565936  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.565938  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.565941  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.565944  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.565947  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.565950  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.565955: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.570583  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.570599  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.570603  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.570606  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.570608  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.570611  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.570614  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.570617  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.570620  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.570623  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.570654: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.575372  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.575393  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.575396  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.575399  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.575402  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.575406  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.575409  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.575411  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.575414  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.575418  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.575423: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.580087  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.580103  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.580106  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.580109  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.580112  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.580115  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.580118  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.580121  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.580124  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.580127  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.580132: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.584687  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.584704  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.584708  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.584711  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.584713  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.584716  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.584719  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.584722  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.584725  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.584728  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.584732: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.589259  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.589275  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.589278  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.589281  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.589284  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.589287  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.589290  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.589293  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.589295  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.589298  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.589303: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.593663  647952 buffer_comparator.cc:145] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1748038445.593678  647952 buffer_comparator.cc:145] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1748038445.593681  647952 buffer_comparator.cc:145] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1748038445.593684  647952 buffer_comparator.cc:145] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1748038445.593687  647952 buffer_comparator.cc:145] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1748038445.593690  647952 buffer_comparator.cc:145] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1748038445.593693  647952 buffer_comparator.cc:145] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1748038445.593696  647952 buffer_comparator.cc:145] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1748038445.593698  647952 buffer_comparator.cc:145] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1748038445.593701  647952 buffer_comparator.cc:145] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-05-23 22:14:05.593706: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.600947  647952 buffer_comparator.cc:145] Difference at 16: 0.2145, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1748038445.600963  647952 buffer_comparator.cc:145] Difference at 17: 0.321023, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1748038445.600967  647952 buffer_comparator.cc:145] Difference at 18: 0.83293, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1748038445.600970  647952 buffer_comparator.cc:145] Difference at 19: 0.829562, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1748038445.600973  647952 buffer_comparator.cc:145] Difference at 20: 0.622037, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1748038445.600976  647952 buffer_comparator.cc:145] Difference at 21: 0.822182, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1748038445.600979  647952 buffer_comparator.cc:145] Difference at 22: 0.161397, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1748038445.600982  647952 buffer_comparator.cc:145] Difference at 23: 0.570683, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1748038445.600985  647952 buffer_comparator.cc:145] Difference at 24: 0.600241, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1748038445.600987  647952 buffer_comparator.cc:145] Difference at 25: 0.57183, expected 357.082</span></span>
<span class="line"><span>2025-05-23 22:14:05.600992: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.604549  647952 buffer_comparator.cc:145] Difference at 16: 0.2145, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1748038445.604565  647952 buffer_comparator.cc:145] Difference at 17: 0.321023, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1748038445.604568  647952 buffer_comparator.cc:145] Difference at 18: 0.83293, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1748038445.604571  647952 buffer_comparator.cc:145] Difference at 19: 0.829562, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1748038445.604574  647952 buffer_comparator.cc:145] Difference at 20: 0.622037, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1748038445.604577  647952 buffer_comparator.cc:145] Difference at 21: 0.822182, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1748038445.604580  647952 buffer_comparator.cc:145] Difference at 22: 0.161397, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1748038445.604583  647952 buffer_comparator.cc:145] Difference at 23: 0.570683, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1748038445.604586  647952 buffer_comparator.cc:145] Difference at 24: 0.600241, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1748038445.604589  647952 buffer_comparator.cc:145] Difference at 25: 0.57183, expected 357.082</span></span>
<span class="line"><span>2025-05-23 22:14:05.604593: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038445.608793  647952 buffer_comparator.cc:145] Difference at 16: 0.2145, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1748038445.608810  647952 buffer_comparator.cc:145] Difference at 17: 0.321023, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1748038445.608814  647952 buffer_comparator.cc:145] Difference at 18: 0.83293, expected 358.37</span></span>
<span class="line"><span>Epoch   1	Train Loss: 15.530720	Train Acc: 14.2857%	Val Loss: 13.579977	Val Acc: 6.6000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 16.045889	Train Acc: 12.1429%	Val Loss: 14.112863	Val Acc: 9.0000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 16.677082	Train Acc: 9.2857%	Val Loss: 15.805104	Val Acc: 10.2000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 16.700085	Train Acc: 9.2857%	Val Loss: 17.111614	Val Acc: 13.0000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 21.159563	Train Acc: 7.8571%	Val Loss: 18.136404	Val Acc: 13.6000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 22.374271	Train Acc: 5.7143%	Val Loss: 19.135838	Val Acc: 14.4000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 23.574816	Train Acc: 3.5714%	Val Loss: 20.127844	Val Acc: 14.2000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 23.169682	Train Acc: 5.7143%	Val Loss: 20.966925	Val Acc: 15.6000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 25.039701	Train Acc: 7.1429%	Val Loss: 21.889334	Val Acc: 16.4000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 31.183416	Train Acc: 8.5714%	Val Loss: 24.022121	Val Acc: 13.0000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 40.457115	Train Acc: 12.8571%	Val Loss: 28.945467	Val Acc: 11.8000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 31.817516	Train Acc: 13.5714%	Val Loss: 34.764469	Val Acc: 12.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 39.263451	Train Acc: 14.2857%	Val Loss: 41.009556	Val Acc: 11.6000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 56.616253	Train Acc: 14.2857%	Val Loss: 47.109940	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 56.070484	Train Acc: 14.2857%	Val Loss: 53.183479	Val Acc: 11.6000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 46.269859	Train Acc: 14.2857%	Val Loss: 58.907463	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 78.278404	Train Acc: 14.2857%	Val Loss: 64.537300	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 57.343861	Train Acc: 14.2857%	Val Loss: 69.991592	Val Acc: 11.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 78.292610	Train Acc: 14.2857%	Val Loss: 75.233971	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 75.226761	Train Acc: 14.2857%	Val Loss: 80.310646	Val Acc: 11.4000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 116.914391	Train Acc: 14.2857%	Val Loss: 85.265656	Val Acc: 11.4000%</span></span>
<span class="line"><span>Early Stopping at Epoch 21</span></span>
<span class="line"><span>2025-05-23 22:15:17.892463: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:15:18.506583: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:15:18.639127: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:15:19.219831: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_31&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1748038519.226678  647952 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748038519.226720  647952 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748038519.226724  647952 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748038519.226727  647952 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.226731  647952 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748038519.226734  647952 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748038519.226737  647952 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748038519.226740  647952 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748038519.226743  647952 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748038519.226746  647952 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-23 22:15:19.226754: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.230068  647952 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748038519.230082  647952 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748038519.230085  647952 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748038519.230088  647952 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.230092  647952 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748038519.230095  647952 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748038519.230098  647952 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748038519.230101  647952 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748038519.230104  647952 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748038519.230107  647952 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-23 22:15:19.230112: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.233441  647952 buffer_comparator.cc:145] Difference at 112: 304.19, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748038519.233455  647952 buffer_comparator.cc:145] Difference at 113: 215.63, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748038519.233458  647952 buffer_comparator.cc:145] Difference at 114: 345.751, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748038519.233461  647952 buffer_comparator.cc:145] Difference at 115: 254.606, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.233464  647952 buffer_comparator.cc:145] Difference at 116: 181.189, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748038519.233467  647952 buffer_comparator.cc:145] Difference at 117: 228.353, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748038519.233472  647952 buffer_comparator.cc:145] Difference at 118: 351.051, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748038519.233475  647952 buffer_comparator.cc:145] Difference at 119: 304.235, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748038519.233478  647952 buffer_comparator.cc:145] Difference at 120: 216.539, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748038519.233481  647952 buffer_comparator.cc:145] Difference at 121: 345.609, expected 1820.16</span></span>
<span class="line"><span>2025-05-23 22:15:19.233486: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.236738  647952 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748038519.236751  647952 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748038519.236754  647952 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748038519.236757  647952 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.236761  647952 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748038519.236764  647952 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748038519.236767  647952 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748038519.236770  647952 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748038519.236773  647952 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748038519.236776  647952 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-23 22:15:19.236781: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.240091  647952 buffer_comparator.cc:145] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1748038519.240104  647952 buffer_comparator.cc:145] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1748038519.240107  647952 buffer_comparator.cc:145] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1748038519.240110  647952 buffer_comparator.cc:145] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.240113  647952 buffer_comparator.cc:145] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1748038519.240117  647952 buffer_comparator.cc:145] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1748038519.240120  647952 buffer_comparator.cc:145] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1748038519.240123  647952 buffer_comparator.cc:145] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1748038519.240126  647952 buffer_comparator.cc:145] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1748038519.240129  647952 buffer_comparator.cc:145] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-05-23 22:15:19.240134: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.243365  647952 buffer_comparator.cc:145] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1748038519.243378  647952 buffer_comparator.cc:145] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748038519.243381  647952 buffer_comparator.cc:145] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1748038519.243384  647952 buffer_comparator.cc:145] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1748038519.243387  647952 buffer_comparator.cc:145] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.243390  647952 buffer_comparator.cc:145] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.243393  647952 buffer_comparator.cc:145] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1748038519.243396  647952 buffer_comparator.cc:145] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1748038519.243400  647952 buffer_comparator.cc:145] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1748038519.243403  647952 buffer_comparator.cc:145] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-05-23 22:15:19.243408: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.246739  647952 buffer_comparator.cc:145] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1748038519.246753  647952 buffer_comparator.cc:145] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1748038519.246757  647952 buffer_comparator.cc:145] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1748038519.246760  647952 buffer_comparator.cc:145] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.246763  647952 buffer_comparator.cc:145] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.246766  647952 buffer_comparator.cc:145] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1748038519.246769  647952 buffer_comparator.cc:145] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1748038519.246772  647952 buffer_comparator.cc:145] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1748038519.246775  647952 buffer_comparator.cc:145] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1748038519.246778  647952 buffer_comparator.cc:145] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.246783: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.249967  647952 buffer_comparator.cc:145] Difference at 0: 1084.56, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.249980  647952 buffer_comparator.cc:145] Difference at 1: 1350.61, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.249984  647952 buffer_comparator.cc:145] Difference at 2: 2009.8, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.249987  647952 buffer_comparator.cc:145] Difference at 3: 1768.04, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.249990  647952 buffer_comparator.cc:145] Difference at 4: 1240.61, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.249993  647952 buffer_comparator.cc:145] Difference at 6: 1407.03, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.249996  647952 buffer_comparator.cc:145] Difference at 7: 1138.83, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.249999  647952 buffer_comparator.cc:145] Difference at 8: 1417.44, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.250002  647952 buffer_comparator.cc:145] Difference at 9: 2084.44, expected 1833.77</span></span>
<span class="line"><span>E0000 00:00:1748038519.250005  647952 buffer_comparator.cc:145] Difference at 10: 1844.73, expected 1592.38</span></span>
<span class="line"><span>2025-05-23 22:15:19.250010: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.253348  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.253362  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.253365  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.253368  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.253371  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.253374  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.253377  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.253380  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.253383  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.253386  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.253393: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.256665  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.256679  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.256683  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.256686  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.256689  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.256692  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.256695  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.256698  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.256701  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.256704  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.256709: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.259980  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.259993  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.259996  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.259999  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.260002  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.260006  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.260009  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.260012  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.260015  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.260018  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.260023: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.263217  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.263234  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.263238  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.263241  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.263244  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.263247  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.263250  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.263253  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.263256  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.263259  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.263265: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.266533  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.266551  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.266555  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.266558  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.266561  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.266565  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.266568  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.266571  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.266574  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.266577  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.266582: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.269838  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.269855  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.269858  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.269861  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.269864  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.269867  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.269870  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.269873  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.269876  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.269880  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.269885: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.273125  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.273141  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.273145  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.273148  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.273151  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.273154  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.273157  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.273160  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.273163  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.273166  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.273171: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.276416  647952 buffer_comparator.cc:145] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1748038519.276436  647952 buffer_comparator.cc:145] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1748038519.276439  647952 buffer_comparator.cc:145] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.276442  647952 buffer_comparator.cc:145] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1748038519.276445  647952 buffer_comparator.cc:145] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.276448  647952 buffer_comparator.cc:145] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1748038519.276451  647952 buffer_comparator.cc:145] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1748038519.276454  647952 buffer_comparator.cc:145] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1748038519.276457  647952 buffer_comparator.cc:145] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1748038519.276460  647952 buffer_comparator.cc:145] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-05-23 22:15:19.276465: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.279761  647952 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.279781  647952 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.279784  647952 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.279787  647952 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.279790  647952 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.279793  647952 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.279796  647952 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.279800  647952 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.279803  647952 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.279806  647952 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-23 22:15:19.279810: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.283081  647952 buffer_comparator.cc:145] Difference at 0: 1100.47, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.283098  647952 buffer_comparator.cc:145] Difference at 1: 1361.33, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.283101  647952 buffer_comparator.cc:145] Difference at 2: 2059.82, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.283104  647952 buffer_comparator.cc:145] Difference at 3: 1808.05, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.283107  647952 buffer_comparator.cc:145] Difference at 4: 1265.06, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.283110  647952 buffer_comparator.cc:145] Difference at 5: 1986, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.283113  647952 buffer_comparator.cc:145] Difference at 6: 1409.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.283116  647952 buffer_comparator.cc:145] Difference at 7: 1173.38, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.283119  647952 buffer_comparator.cc:145] Difference at 8: 1420.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.283122  647952 buffer_comparator.cc:145] Difference at 9: 2114.57, expected 1833.77</span></span>
<span class="line"><span>2025-05-23 22:15:19.283127: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.286304  647952 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.286318  647952 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.286321  647952 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.286325  647952 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.286328  647952 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.286331  647952 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.286335  647952 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.286338  647952 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.286341  647952 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.286344  647952 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-23 22:15:19.286349: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.289592  647952 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.289607  647952 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.289610  647952 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.289613  647952 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.289616  647952 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.289619  647952 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.289622  647952 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.289625  647952 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.289653  647952 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.289657  647952 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-23 22:15:19.289662: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.292969  647952 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.292988  647952 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.292991  647952 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.292994  647952 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.292997  647952 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.293000  647952 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.293003  647952 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.293006  647952 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.293009  647952 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.293012  647952 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-23 22:15:19.293018: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.296365  647952 buffer_comparator.cc:145] Difference at 0: 1134.52, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.296386  647952 buffer_comparator.cc:145] Difference at 1: 1372.21, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.296389  647952 buffer_comparator.cc:145] Difference at 2: 2061.12, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.296392  647952 buffer_comparator.cc:145] Difference at 3: 1819.93, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.296395  647952 buffer_comparator.cc:145] Difference at 4: 1282.86, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.296398  647952 buffer_comparator.cc:145] Difference at 5: 1994.92, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.296403  647952 buffer_comparator.cc:145] Difference at 6: 1439.18, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.296406  647952 buffer_comparator.cc:145] Difference at 7: 1151.8, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.296409  647952 buffer_comparator.cc:145] Difference at 8: 1413.9, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.296412  647952 buffer_comparator.cc:145] Difference at 9: 2069.06, expected 1833.77</span></span>
<span class="line"><span>2025-05-23 22:15:19.296418: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.299722  647952 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748038519.299738  647952 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.299742  647952 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1748038519.299745  647952 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748038519.299748  647952 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1748038519.299751  647952 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748038519.299754  647952 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1748038519.299757  647952 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.299760  647952 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1748038519.299763  647952 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-23 22:15:19.299768: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.303198  647952 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.303213  647952 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748038519.303217  647952 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1748038519.303220  647952 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1748038519.303224  647952 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1748038519.303227  647952 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.303231  647952 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748038519.303234  647952 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1748038519.303238  647952 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1748038519.303241  647952 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-23 22:15:19.303246: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.306435  647952 buffer_comparator.cc:145] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.306452  647952 buffer_comparator.cc:145] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1748038519.306456  647952 buffer_comparator.cc:145] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1748038519.306460  647952 buffer_comparator.cc:145] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1748038519.306463  647952 buffer_comparator.cc:145] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1748038519.306467  647952 buffer_comparator.cc:145] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1748038519.306470  647952 buffer_comparator.cc:145] Difference at 521: 1727, expected 1550.38</span></span>
<span class="line"><span>E0000 00:00:1748038519.306473  647952 buffer_comparator.cc:145] Difference at 522: 1234.11, expected 1093.77</span></span>
<span class="line"><span>E0000 00:00:1748038519.306478  647952 buffer_comparator.cc:145] Difference at 620: 1248.38, expected 1121.09</span></span>
<span class="line"><span>E0000 00:00:1748038519.306481  647952 buffer_comparator.cc:145] Difference at 690: 1252.32, expected 1120.62</span></span>
<span class="line"><span>2025-05-23 22:15:19.306487: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.309810  647952 buffer_comparator.cc:145] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1748038519.309829  647952 buffer_comparator.cc:145] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.309833  647952 buffer_comparator.cc:145] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1748038519.309836  647952 buffer_comparator.cc:145] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1748038519.309839  647952 buffer_comparator.cc:145] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1748038519.309842  647952 buffer_comparator.cc:145] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1748038519.309845  647952 buffer_comparator.cc:145] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1748038519.309848  647952 buffer_comparator.cc:145] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.309851  647952 buffer_comparator.cc:145] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1748038519.309854  647952 buffer_comparator.cc:145] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-05-23 22:15:19.309859: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038519.313233  647952 buffer_comparator.cc:145] Difference at 0: 1144.96, expected 928.598</span></span>
<span class="line"><span>E0000 00:00:1748038519.313248  647952 buffer_comparator.cc:145] Difference at 1: 1334.45, expected 1186.9</span></span>
<span class="line"><span>E0000 00:00:1748038519.313251  647952 buffer_comparator.cc:145] Difference at 2: 2071.77, expected 1796.78</span></span>
<span class="line"><span>E0000 00:00:1748038519.313254  647952 buffer_comparator.cc:145] Difference at 3: 1855.89, expected 1565.85</span></span>
<span class="line"><span>E0000 00:00:1748038519.313257  647952 buffer_comparator.cc:145] Difference at 4: 1308.71, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1748038519.313260  647952 buffer_comparator.cc:145] Difference at 5: 2021.12, expected 1757.8</span></span>
<span class="line"><span>E0000 00:00:1748038519.313263  647952 buffer_comparator.cc:145] Difference at 6: 1417.87, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1748038519.313266  647952 buffer_comparator.cc:145] Difference at 7: 1204.51, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1748038519.313269  647952 buffer_comparator.cc:145] Difference at 8: 1401.77, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1748038519.313272  647952 buffer_comparator.cc:145] Difference at 9: 2107.26, expected 1833.77</span></span>
<span class="line"><span>2025-05-23 22:15:19.313277: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-05-23 22:15:21.834469: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:15:22.724250: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 200 bytes spill stores, 200 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:15:22.916249: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-05-23 22:15:23.055748: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_27&#39;, 348 bytes spill stores, 348 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1748038523.062308  647952 buffer_comparator.cc:145] Difference at 112: -195.006, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748038523.062367  647952 buffer_comparator.cc:145] Difference at 113: -223.714, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748038523.062383  647952 buffer_comparator.cc:145] Difference at 114: -262.494, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748038523.062387  647952 buffer_comparator.cc:145] Difference at 115: -237.841, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748038523.062390  647952 buffer_comparator.cc:145] Difference at 116: -200.804, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748038523.062393  647952 buffer_comparator.cc:145] Difference at 117: -156.129, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748038523.062396  647952 buffer_comparator.cc:145] Difference at 118: -276.104, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748038523.062399  647952 buffer_comparator.cc:145] Difference at 119: -132.389, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748038523.062403  647952 buffer_comparator.cc:145] Difference at 120: -146.607, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748038523.062406  647952 buffer_comparator.cc:145] Difference at 121: -179.631, expected 1044.63</span></span>
<span class="line"><span>2025-05-23 22:15:23.062418: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.065736  647952 buffer_comparator.cc:145] Difference at 112: -195.006, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748038523.065751  647952 buffer_comparator.cc:145] Difference at 113: -223.714, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748038523.065755  647952 buffer_comparator.cc:145] Difference at 114: -262.494, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748038523.065758  647952 buffer_comparator.cc:145] Difference at 115: -237.841, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748038523.065761  647952 buffer_comparator.cc:145] Difference at 116: -200.804, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748038523.065764  647952 buffer_comparator.cc:145] Difference at 117: -156.129, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748038523.065767  647952 buffer_comparator.cc:145] Difference at 118: -276.104, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748038523.065771  647952 buffer_comparator.cc:145] Difference at 119: -132.389, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748038523.065774  647952 buffer_comparator.cc:145] Difference at 120: -146.607, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748038523.065777  647952 buffer_comparator.cc:145] Difference at 121: -179.631, expected 1044.63</span></span>
<span class="line"><span>2025-05-23 22:15:23.065782: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.069098  647952 buffer_comparator.cc:145] Difference at 112: -195.006, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748038523.069110  647952 buffer_comparator.cc:145] Difference at 113: -223.714, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748038523.069114  647952 buffer_comparator.cc:145] Difference at 114: -262.494, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748038523.069117  647952 buffer_comparator.cc:145] Difference at 115: -237.841, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748038523.069120  647952 buffer_comparator.cc:145] Difference at 116: -200.804, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748038523.069124  647952 buffer_comparator.cc:145] Difference at 117: -156.129, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748038523.069127  647952 buffer_comparator.cc:145] Difference at 118: -276.104, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748038523.069130  647952 buffer_comparator.cc:145] Difference at 119: -132.389, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748038523.069133  647952 buffer_comparator.cc:145] Difference at 120: -146.607, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748038523.069136  647952 buffer_comparator.cc:145] Difference at 121: -179.631, expected 1044.63</span></span>
<span class="line"><span>2025-05-23 22:15:23.069141: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.072335  647952 buffer_comparator.cc:145] Difference at 112: 0.010274, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748038523.072348  647952 buffer_comparator.cc:145] Difference at 113: -0.0197475, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748038523.072353  647952 buffer_comparator.cc:145] Difference at 114: -0.0706815, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748038523.072356  647952 buffer_comparator.cc:145] Difference at 115: 0.0490984, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748038523.072361  647952 buffer_comparator.cc:145] Difference at 116: 0.000698235, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748038523.072365  647952 buffer_comparator.cc:145] Difference at 117: 0.0598805, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748038523.072368  647952 buffer_comparator.cc:145] Difference at 118: 0.0256289, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748038523.072371  647952 buffer_comparator.cc:145] Difference at 119: -0.0374779, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748038523.072374  647952 buffer_comparator.cc:145] Difference at 120: 0.0141777, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748038523.072377  647952 buffer_comparator.cc:145] Difference at 121: 0.0116012, expected 1044.63</span></span>
<span class="line"><span>2025-05-23 22:15:23.072382: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.075720  647952 buffer_comparator.cc:145] Difference at 112: 0.010274, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1748038523.075733  647952 buffer_comparator.cc:145] Difference at 113: -0.0197475, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1748038523.075738  647952 buffer_comparator.cc:145] Difference at 114: -0.0706815, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1748038523.075741  647952 buffer_comparator.cc:145] Difference at 115: 0.0490984, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1748038523.075744  647952 buffer_comparator.cc:145] Difference at 116: 0.000698235, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1748038523.075748  647952 buffer_comparator.cc:145] Difference at 117: 0.0598805, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1748038523.075751  647952 buffer_comparator.cc:145] Difference at 118: 0.0256289, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1748038523.075754  647952 buffer_comparator.cc:145] Difference at 119: -0.0374779, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1748038523.075757  647952 buffer_comparator.cc:145] Difference at 120: 0.0141777, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1748038523.075760  647952 buffer_comparator.cc:145] Difference at 121: 0.0116012, expected 1044.63</span></span>
<span class="line"><span>2025-05-23 22:15:23.075765: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.079010  647952 buffer_comparator.cc:145] Difference at 224: 0.0023712, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1748038523.079022  647952 buffer_comparator.cc:145] Difference at 225: 0.0326954, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1748038523.079027  647952 buffer_comparator.cc:145] Difference at 226: 0.0602755, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1748038523.079030  647952 buffer_comparator.cc:145] Difference at 227: 0.0250077, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1748038523.079033  647952 buffer_comparator.cc:145] Difference at 228: 0.0217244, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1748038523.079036  647952 buffer_comparator.cc:145] Difference at 229: -0.0652699, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1748038523.079039  647952 buffer_comparator.cc:145] Difference at 230: -0.0613384, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1748038523.079042  647952 buffer_comparator.cc:145] Difference at 231: -0.0438705, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1748038523.079045  647952 buffer_comparator.cc:145] Difference at 232: 0.0316087, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1748038523.079048  647952 buffer_comparator.cc:145] Difference at 233: -0.074999, expected 1029.02</span></span>
<span class="line"><span>2025-05-23 22:15:23.079054: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.082385  647952 buffer_comparator.cc:145] Difference at 224: 0.0147585, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1748038523.082398  647952 buffer_comparator.cc:145] Difference at 225: 0.0797445, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1748038523.082403  647952 buffer_comparator.cc:145] Difference at 226: 0.000391293, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1748038523.082406  647952 buffer_comparator.cc:145] Difference at 227: 0.0179735, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1748038523.082409  647952 buffer_comparator.cc:145] Difference at 228: -0.0452672, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1748038523.082414  647952 buffer_comparator.cc:145] Difference at 229: -0.095104, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1748038523.082417  647952 buffer_comparator.cc:145] Difference at 230: -0.0616802, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1748038523.082420  647952 buffer_comparator.cc:145] Difference at 231: -0.0605929, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1748038523.082423  647952 buffer_comparator.cc:145] Difference at 232: -0.0101051, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1748038523.082426  647952 buffer_comparator.cc:145] Difference at 233: -0.0264666, expected 1029.02</span></span>
<span class="line"><span>2025-05-23 22:15:23.082431: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.085638  647952 buffer_comparator.cc:145] Difference at 0: 903.336, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748038523.085654  647952 buffer_comparator.cc:145] Difference at 1: 1271.45, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748038523.085658  647952 buffer_comparator.cc:145] Difference at 2: 1218.72, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748038523.085661  647952 buffer_comparator.cc:145] Difference at 3: 1830.29, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748038523.085664  647952 buffer_comparator.cc:145] Difference at 4: 1832.52, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748038523.085668  647952 buffer_comparator.cc:145] Difference at 5: 1505.57, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748038523.085671  647952 buffer_comparator.cc:145] Difference at 6: 1003.78, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748038523.085674  647952 buffer_comparator.cc:145] Difference at 7: 895.724, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748038523.085678  647952 buffer_comparator.cc:145] Difference at 8: 1254.14, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748038523.085681  647952 buffer_comparator.cc:145] Difference at 9: 1207.96, expected 1052.46</span></span>
<span class="line"><span>2025-05-23 22:15:23.085686: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.089051  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.089063  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.089067  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.089070  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.089074  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.089077  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.089080  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.089083  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.089086  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.089089  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.089094: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.092354  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.092366  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.092370  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.092373  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.092377  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.092380  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.092383  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.092388  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.092391  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.092394  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.092400: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.095650  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.095662  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.095666  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.095670  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.095673  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.095676  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.095679  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.095682  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.095685  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.095688  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.095693: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.098858  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.098869  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.098873  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.098876  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.098880  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.098883  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.098886  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.098889  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.098892  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.098895  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.098900: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.102134  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.102148  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.102152  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.102155  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.102159  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.102162  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.102165  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.102168  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.102172  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.102175  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.102180: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.105422  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.105434  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.105438  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.105441  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.105445  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.105448  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.105451  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.105454  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.105457  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.105460  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.105465: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.108687  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.108699  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.108703  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.108706  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.108709  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.108713  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.108716  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.108719  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.108722  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.108725  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.108730: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.111922  647952 buffer_comparator.cc:145] Difference at 448: 0.0217227, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1748038523.111935  647952 buffer_comparator.cc:145] Difference at 449: -0.0353118, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1748038523.111939  647952 buffer_comparator.cc:145] Difference at 450: 0.000868525, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1748038523.111942  647952 buffer_comparator.cc:145] Difference at 451: 0.0119535, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1748038523.111945  647952 buffer_comparator.cc:145] Difference at 452: -0.107458, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1748038523.111948  647952 buffer_comparator.cc:145] Difference at 453: -0.042944, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1748038523.111951  647952 buffer_comparator.cc:145] Difference at 454: -0.106297, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1748038523.111954  647952 buffer_comparator.cc:145] Difference at 455: -0.0341475, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1748038523.111958  647952 buffer_comparator.cc:145] Difference at 456: 0.0454307, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1748038523.111962  647952 buffer_comparator.cc:145] Difference at 457: 0.00894434, expected 1051.03</span></span>
<span class="line"><span>2025-05-23 22:15:23.111967: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.115207  647952 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748038523.115220  647952 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748038523.115224  647952 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748038523.115227  647952 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748038523.115230  647952 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748038523.115233  647952 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748038523.115236  647952 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748038523.115240  647952 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748038523.115243  647952 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1748038523.115246  647952 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-05-23 22:15:23.115251: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.118461  647952 buffer_comparator.cc:145] Difference at 0: 876.475, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748038523.118473  647952 buffer_comparator.cc:145] Difference at 1: 1292.4, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748038523.118477  647952 buffer_comparator.cc:145] Difference at 2: 1239.8, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748038523.118480  647952 buffer_comparator.cc:145] Difference at 3: 1830.71, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748038523.118483  647952 buffer_comparator.cc:145] Difference at 4: 1857.47, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748038523.118486  647952 buffer_comparator.cc:145] Difference at 5: 1551.94, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748038523.118489  647952 buffer_comparator.cc:145] Difference at 6: 1022.45, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748038523.118493  647952 buffer_comparator.cc:145] Difference at 8: 1214.29, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748038523.118496  647952 buffer_comparator.cc:145] Difference at 9: 1173.34, expected 1052.46</span></span>
<span class="line"><span>E0000 00:00:1748038523.118499  647952 buffer_comparator.cc:145] Difference at 10: 1732.94, expected 1556.04</span></span>
<span class="line"><span>2025-05-23 22:15:23.118504: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.121654  647952 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748038523.121666  647952 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748038523.121670  647952 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748038523.121673  647952 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748038523.121676  647952 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748038523.121679  647952 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748038523.121683  647952 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748038523.121686  647952 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748038523.121689  647952 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748038523.121692  647952 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-23 22:15:23.121697: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.124924  647952 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748038523.124936  647952 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748038523.124940  647952 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748038523.124943  647952 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748038523.124946  647952 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748038523.124950  647952 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748038523.124953  647952 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748038523.124956  647952 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748038523.124959  647952 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748038523.124962  647952 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-23 22:15:23.124967: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.128225  647952 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748038523.128237  647952 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748038523.128240  647952 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748038523.128244  647952 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748038523.128247  647952 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748038523.128250  647952 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748038523.128253  647952 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748038523.128256  647952 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748038523.128259  647952 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748038523.128262  647952 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-23 22:15:23.128267: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.131549  647952 buffer_comparator.cc:145] Difference at 0: 885.439, expected 761.386</span></span>
<span class="line"><span>E0000 00:00:1748038523.131561  647952 buffer_comparator.cc:145] Difference at 1: 1301.12, expected 1085.49</span></span>
<span class="line"><span>E0000 00:00:1748038523.131565  647952 buffer_comparator.cc:145] Difference at 2: 1218.31, expected 1038.52</span></span>
<span class="line"><span>E0000 00:00:1748038523.131568  647952 buffer_comparator.cc:145] Difference at 3: 1809.83, expected 1537.78</span></span>
<span class="line"><span>E0000 00:00:1748038523.131571  647952 buffer_comparator.cc:145] Difference at 4: 1850.11, expected 1560.75</span></span>
<span class="line"><span>E0000 00:00:1748038523.131574  647952 buffer_comparator.cc:145] Difference at 5: 1525.04, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1748038523.131578  647952 buffer_comparator.cc:145] Difference at 6: 1032.03, expected 863.844</span></span>
<span class="line"><span>E0000 00:00:1748038523.131581  647952 buffer_comparator.cc:145] Difference at 7: 857.478, expected 765.38</span></span>
<span class="line"><span>E0000 00:00:1748038523.131584  647952 buffer_comparator.cc:145] Difference at 8: 1259.9, expected 1089.92</span></span>
<span class="line"><span>E0000 00:00:1748038523.131587  647952 buffer_comparator.cc:145] Difference at 9: 1198.08, expected 1052.46</span></span>
<span class="line"><span>2025-05-23 22:15:23.131592: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038523.134854  647952 buffer_comparator.cc:145] Difference at 896: 0.0065422, expected 767.869</span></span>
<span class="line"><span>Test Loss: 81.086784	Test Acc: 10.3000%</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
