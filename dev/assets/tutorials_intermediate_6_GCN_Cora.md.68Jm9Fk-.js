import{_ as s,c as n,o as e,al as p}from"./chunks/framework.BCN3FD2k.js";const E=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-04-05 15:06:48.480987: I external/xla/xla/service/service.cc:152] XLA service 0x6317130 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-05 15:06:48.481121: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743865608.482345  183376 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743865608.482522  183376 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743865608.482629  183376 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743865608.498940  183376 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-3/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-3/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-04-05 15:07:51.193448: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:52.001126: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:52.934766: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:52.997041: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 268 bytes spill stores, 268 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:53.758503: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:53.799744: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 616 bytes spill stores, 616 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.126175: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.150248: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.381606: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 52 bytes spill stores, 52 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.426520: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.516590: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.523161: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.618180: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:54.964064: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:55.778809: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:56.171628: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:56.440308: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:56.655592: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 256 bytes spill stores, 256 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:07:56.925849: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743865677.073473  183376 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.074621  183376 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743865677.074630  183376 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743865677.074638  183376 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743865677.074644  183376 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743865677.074651  183376 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743865677.074658  183376 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865677.074665  183376 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743865677.074672  183376 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743865677.074679  183376 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.074694: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.078422  183376 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.078456  183376 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743865677.078464  183376 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743865677.078471  183376 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743865677.078478  183376 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743865677.078484  183376 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743865677.078491  183376 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865677.078498  183376 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743865677.078505  183376 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743865677.078512  183376 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.078522: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.081881  183376 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.081911  183376 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743865677.081918  183376 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743865677.081925  183376 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743865677.081932  183376 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743865677.081939  183376 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743865677.081946  183376 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865677.081953  183376 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743865677.081959  183376 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743865677.081966  183376 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.081979: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.085446  183376 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.085479  183376 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743865677.085486  183376 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743865677.085493  183376 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743865677.085500  183376 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743865677.085507  183376 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743865677.085514  183376 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865677.085521  183376 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743865677.085528  183376 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743865677.085534  183376 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.085545: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.088926  183376 buffer_comparator.cc:156] Difference at 0: 1140.51, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743865677.088940  183376 buffer_comparator.cc:156] Difference at 1: 1405.76, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743865677.088943  183376 buffer_comparator.cc:156] Difference at 2: 2133.59, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743865677.088946  183376 buffer_comparator.cc:156] Difference at 3: 1840.1, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743865677.088949  183376 buffer_comparator.cc:156] Difference at 4: 1308.3, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743865677.088952  183376 buffer_comparator.cc:156] Difference at 5: 2065.73, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.088955  183376 buffer_comparator.cc:156] Difference at 6: 1481.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1743865677.088958  183376 buffer_comparator.cc:156] Difference at 7: 1113.93, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743865677.088961  183376 buffer_comparator.cc:156] Difference at 8: 1359.57, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.088964  183376 buffer_comparator.cc:156] Difference at 9: 2049.53, expected 1833.76</span></span>
<span class="line"><span>2025-04-05 15:07:57.088969: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.092089  183376 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.092103  183376 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743865677.092106  183376 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743865677.092109  183376 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743865677.092112  183376 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743865677.092115  183376 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743865677.092118  183376 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865677.092121  183376 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743865677.092124  183376 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743865677.092127  183376 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.092132: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.095239  183376 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743865677.095253  183376 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743865677.095256  183376 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743865677.095259  183376 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743865677.095262  183376 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743865677.095265  183376 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743865677.095268  183376 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743865677.095271  183376 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743865677.095274  183376 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743865677.095277  183376 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-04-05 15:07:57.095282: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.098326  183376 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743865677.098339  183376 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743865677.098342  183376 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743865677.098346  183376 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743865677.098349  183376 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743865677.098352  183376 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743865677.098355  183376 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743865677.098358  183376 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743865677.098361  183376 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743865677.098364  183376 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-04-05 15:07:57.098369: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.101453  183376 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743865677.101467  183376 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743865677.101470  183376 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743865677.101473  183376 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743865677.101476  183376 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.101479  183376 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.101482  183376 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743865677.101485  183376 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743865677.101488  183376 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743865677.101491  183376 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-04-05 15:07:57.101496: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.104696  183376 buffer_comparator.cc:156] Difference at 0: 1057.99, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743865677.104710  183376 buffer_comparator.cc:156] Difference at 1: 1320.07, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743865677.104714  183376 buffer_comparator.cc:156] Difference at 2: 2005.81, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743865677.104717  183376 buffer_comparator.cc:156] Difference at 3: 1746.91, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743865677.104720  183376 buffer_comparator.cc:156] Difference at 4: 1253.09, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743865677.104723  183376 buffer_comparator.cc:156] Difference at 7: 1176.41, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743865677.104726  183376 buffer_comparator.cc:156] Difference at 8: 1399.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.104729  183376 buffer_comparator.cc:156] Difference at 9: 2127.08, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1743865677.104732  183376 buffer_comparator.cc:156] Difference at 10: 1879.74, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1743865677.104735  183376 buffer_comparator.cc:156] Difference at 11: 1363.62, expected 1121.95</span></span>
<span class="line"><span>2025-04-05 15:07:57.104740: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.107829  183376 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743865677.107843  183376 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743865677.107846  183376 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743865677.107849  183376 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743865677.107852  183376 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.107855  183376 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.107858  183376 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743865677.107861  183376 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743865677.107864  183376 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743865677.107867  183376 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-04-05 15:07:57.107872: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.110950  183376 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743865677.110963  183376 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743865677.110966  183376 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743865677.110969  183376 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743865677.110972  183376 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.110975  183376 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.110978  183376 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743865677.110981  183376 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743865677.110984  183376 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743865677.110987  183376 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-04-05 15:07:57.110992: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.114171  183376 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743865677.114184  183376 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865677.114187  183376 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.114192  183376 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743865677.114195  183376 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865677.114198  183376 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.114201  183376 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.114204  183376 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743865677.114207  183376 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743865677.114210  183376 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-05 15:07:57.114215: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.117378  183376 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743865677.117392  183376 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865677.117395  183376 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.117398  183376 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743865677.117401  183376 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865677.117404  183376 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.117407  183376 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.117410  183376 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743865677.117413  183376 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743865677.117417  183376 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-05 15:07:57.117421: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.120616  183376 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743865677.120629  183376 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865677.120633  183376 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.120636  183376 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743865677.120639  183376 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865677.120642  183376 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.120645  183376 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.120648  183376 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743865677.120651  183376 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743865677.120654  183376 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-05 15:07:57.120659: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.123743  183376 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743865677.123757  183376 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865677.123760  183376 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.123763  183376 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743865677.123766  183376 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865677.123771  183376 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.123774  183376 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.123777  183376 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743865677.123780  183376 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743865677.123783  183376 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-05 15:07:57.123787: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.126844  183376 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743865677.126859  183376 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865677.126862  183376 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743865677.126865  183376 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743865677.126868  183376 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865677.126871  183376 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.126874  183376 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865677.126877  183376 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743865677.126880  183376 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743865677.126883  183376 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-05 15:07:57.126888: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.130231  183376 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743865677.130245  183376 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743865677.130248  183376 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.130251  183376 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743865677.130254  183376 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743865677.130258  183376 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743865677.130261  183376 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743865677.130264  183376 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743865677.130267  183376 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743865677.130270  183376 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-04-05 15:07:57.130275: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.133646  183376 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743865677.133659  183376 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743865677.133662  183376 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743865677.133666  183376 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743865677.133669  183376 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743865677.133672  183376 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743865677.133675  183376 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743865677.133679  183376 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743865677.133682  183376 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743865677.133685  183376 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-04-05 15:07:57.133690: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.143766  183376 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1743865677.143785  183376 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1743865677.143790  183376 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1743865677.143794  183376 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1743865677.143798  183376 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1743865677.143802  183376 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1743865677.143806  183376 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1743865677.143810  183376 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1743865677.143814  183376 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.143818  183376 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-04-05 15:07:57.143825: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.150985  183376 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1743865677.150998  183376 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1743865677.151001  183376 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1743865677.151004  183376 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1743865677.151007  183376 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1743865677.151010  183376 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1743865677.151013  183376 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1743865677.151016  183376 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1743865677.151019  183376 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.151022  183376 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-04-05 15:07:57.151026: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.155881  183376 buffer_comparator.cc:156] Difference at 64: 0, expected 1106.21</span></span>
<span class="line"><span>E0000 00:00:1743865677.155896  183376 buffer_comparator.cc:156] Difference at 65: 0, expected 1087.83</span></span>
<span class="line"><span>E0000 00:00:1743865677.155899  183376 buffer_comparator.cc:156] Difference at 66: 0, expected 1090.54</span></span>
<span class="line"><span>E0000 00:00:1743865677.155902  183376 buffer_comparator.cc:156] Difference at 67: 0, expected 1104.23</span></span>
<span class="line"><span>E0000 00:00:1743865677.155905  183376 buffer_comparator.cc:156] Difference at 68: 0, expected 1104.3</span></span>
<span class="line"><span>E0000 00:00:1743865677.155908  183376 buffer_comparator.cc:156] Difference at 69: 0, expected 1093.45</span></span>
<span class="line"><span>E0000 00:00:1743865677.155911  183376 buffer_comparator.cc:156] Difference at 70: 0, expected 1091.52</span></span>
<span class="line"><span>E0000 00:00:1743865677.155914  183376 buffer_comparator.cc:156] Difference at 71: 0, expected 1110.4</span></span>
<span class="line"><span>E0000 00:00:1743865677.155917  183376 buffer_comparator.cc:156] Difference at 72: 0, expected 1106.92</span></span>
<span class="line"><span>E0000 00:00:1743865677.155920  183376 buffer_comparator.cc:156] Difference at 73: 0, expected 1088.44</span></span>
<span class="line"><span>2025-04-05 15:07:57.155924: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.160790  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.160805  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.160808  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.160811  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.160814  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.160817  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.160820  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.160823  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.160825  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.160828  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.160833: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.165397  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.165413  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.165416  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.165419  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.165422  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.165425  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.165428  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.165431  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.165434  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.165437  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.165441: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.169969  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.169982  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.169986  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.169989  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.169991  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.169994  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.169997  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.170000  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.170003  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.170006  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.170011: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.174964  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.174978  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.174981  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.174986  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.174989  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.174992  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.174994  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.174997  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.175000  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.175003  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.175008: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.180033  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.180047  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.180051  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.180054  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.180057  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.180060  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.180063  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.180065  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.180068  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.180071  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.180076: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.184687  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.184700  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.184703  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.184706  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.184709  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.184712  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.184715  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.184718  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.184721  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.184724  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.184729: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.189459  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.189473  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.189476  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.189479  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.189482  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.189485  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.189488  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.189492  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.189495  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.189498  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.189502: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.193918  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.193932  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.193935  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.193938  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.193941  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.193944  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.193947  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.193950  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.193952  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.193955  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.193960: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.198740  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.198757  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.198760  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.198763  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.198766  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.198769  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.198772  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.198775  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.198778  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.198781  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.198785: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.203363  183376 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743865677.203377  183376 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743865677.203380  183376 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743865677.203383  183376 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743865677.203386  183376 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743865677.203389  183376 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743865677.203392  183376 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743865677.203395  183376 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743865677.203398  183376 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743865677.203401  183376 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-05 15:07:57.203405: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.207847  183376 buffer_comparator.cc:156] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1743865677.207863  183376 buffer_comparator.cc:156] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1743865677.207866  183376 buffer_comparator.cc:156] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1743865677.207869  183376 buffer_comparator.cc:156] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1743865677.207872  183376 buffer_comparator.cc:156] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1743865677.207874  183376 buffer_comparator.cc:156] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1743865677.207877  183376 buffer_comparator.cc:156] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1743865677.207880  183376 buffer_comparator.cc:156] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1743865677.207883  183376 buffer_comparator.cc:156] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1743865677.207886  183376 buffer_comparator.cc:156] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-04-05 15:07:57.207891: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.210945  183376 buffer_comparator.cc:156] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743865677.210956  183376 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743865677.210960  183376 buffer_comparator.cc:156] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743865677.210962  183376 buffer_comparator.cc:156] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743865677.210965  183376 buffer_comparator.cc:156] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743865677.210968  183376 buffer_comparator.cc:156] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743865677.210971  183376 buffer_comparator.cc:156] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743865677.210974  183376 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743865677.210977  183376 buffer_comparator.cc:156] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743865677.210979  183376 buffer_comparator.cc:156] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-04-05 15:07:57.210984: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.213099  183376 buffer_comparator.cc:156] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743865677.213110  183376 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743865677.213113  183376 buffer_comparator.cc:156] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743865677.213116  183376 buffer_comparator.cc:156] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743865677.213119  183376 buffer_comparator.cc:156] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743865677.213122  183376 buffer_comparator.cc:156] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743865677.213125  183376 buffer_comparator.cc:156] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743865677.213128  183376 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743865677.213130  183376 buffer_comparator.cc:156] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743865677.213133  183376 buffer_comparator.cc:156] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-04-05 15:07:57.213137: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.215249  183376 buffer_comparator.cc:156] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743865677.215261  183376 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743865677.215264  183376 buffer_comparator.cc:156] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743865677.215268  183376 buffer_comparator.cc:156] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743865677.215271  183376 buffer_comparator.cc:156] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743865677.215274  183376 buffer_comparator.cc:156] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743865677.215277  183376 buffer_comparator.cc:156] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743865677.215280  183376 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743865677.215282  183376 buffer_comparator.cc:156] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743865677.215285  183376 buffer_comparator.cc:156] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-04-05 15:07:57.215290: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.217416  183376 buffer_comparator.cc:156] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1743865677.217426  183376 buffer_comparator.cc:156] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1743865677.217429  183376 buffer_comparator.cc:156] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1743865677.217432  183376 buffer_comparator.cc:156] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1743865677.217435  183376 buffer_comparator.cc:156] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1743865677.217438  183376 buffer_comparator.cc:156] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1743865677.217441  183376 buffer_comparator.cc:156] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1743865677.217443  183376 buffer_comparator.cc:156] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1743865677.217446  183376 buffer_comparator.cc:156] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1743865677.217449  183376 buffer_comparator.cc:156] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-04-05 15:07:57.217453: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.219574  183376 buffer_comparator.cc:156] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1743865677.219586  183376 buffer_comparator.cc:156] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1743865677.219589  183376 buffer_comparator.cc:156] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1743865677.219591  183376 buffer_comparator.cc:156] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1743865677.219594  183376 buffer_comparator.cc:156] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1743865677.219597  183376 buffer_comparator.cc:156] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1743865677.219600  183376 buffer_comparator.cc:156] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1743865677.219602  183376 buffer_comparator.cc:156] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1743865677.219605  183376 buffer_comparator.cc:156] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1743865677.219608  183376 buffer_comparator.cc:156] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-04-05 15:07:57.219612: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.221728  183376 buffer_comparator.cc:156] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743865677.221739  183376 buffer_comparator.cc:156] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743865677.221742  183376 buffer_comparator.cc:156] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743865677.221745  183376 buffer_comparator.cc:156] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743865677.221748  183376 buffer_comparator.cc:156] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743865677.221751  183376 buffer_comparator.cc:156] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743865677.221755  183376 buffer_comparator.cc:156] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743865677.221758  183376 buffer_comparator.cc:156] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743865677.221761  183376 buffer_comparator.cc:156] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743865677.221763  183376 buffer_comparator.cc:156] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-05 15:07:57.221768: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.223917  183376 buffer_comparator.cc:156] Difference at 0: 16.5369, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1743865677.223928  183376 buffer_comparator.cc:156] Difference at 1: 19.4176, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1743865677.223931  183376 buffer_comparator.cc:156] Difference at 2: 16.204, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1743865677.223935  183376 buffer_comparator.cc:156] Difference at 6: 13.1759, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1743865677.223938  183376 buffer_comparator.cc:156] Difference at 9: 16.3002, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1743865677.223941  183376 buffer_comparator.cc:156] Difference at 11: 15.6508, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1743865677.223944  183376 buffer_comparator.cc:156] Difference at 12: 20.6885, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1743865677.223947  183376 buffer_comparator.cc:156] Difference at 13: 17.247, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1743865677.223950  183376 buffer_comparator.cc:156] Difference at 14: 14.7694, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1743865677.223953  183376 buffer_comparator.cc:156] Difference at 16: 17.2743, expected 15.1227</span></span>
<span class="line"><span>2025-04-05 15:07:57.223957: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.226097  183376 buffer_comparator.cc:156] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743865677.226108  183376 buffer_comparator.cc:156] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743865677.226111  183376 buffer_comparator.cc:156] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743865677.226114  183376 buffer_comparator.cc:156] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743865677.226117  183376 buffer_comparator.cc:156] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743865677.226120  183376 buffer_comparator.cc:156] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743865677.226122  183376 buffer_comparator.cc:156] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743865677.226125  183376 buffer_comparator.cc:156] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743865677.226128  183376 buffer_comparator.cc:156] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743865677.226131  183376 buffer_comparator.cc:156] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-05 15:07:57.226135: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.228257  183376 buffer_comparator.cc:156] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743865677.228268  183376 buffer_comparator.cc:156] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743865677.228271  183376 buffer_comparator.cc:156] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743865677.228274  183376 buffer_comparator.cc:156] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743865677.228277  183376 buffer_comparator.cc:156] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743865677.228280  183376 buffer_comparator.cc:156] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743865677.228283  183376 buffer_comparator.cc:156] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743865677.228285  183376 buffer_comparator.cc:156] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743865677.228288  183376 buffer_comparator.cc:156] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743865677.228292  183376 buffer_comparator.cc:156] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-05 15:07:57.228297: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.230419  183376 buffer_comparator.cc:156] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743865677.230430  183376 buffer_comparator.cc:156] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743865677.230433  183376 buffer_comparator.cc:156] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743865677.230436  183376 buffer_comparator.cc:156] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743865677.230439  183376 buffer_comparator.cc:156] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743865677.230442  183376 buffer_comparator.cc:156] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743865677.230444  183376 buffer_comparator.cc:156] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743865677.230447  183376 buffer_comparator.cc:156] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743865677.230450  183376 buffer_comparator.cc:156] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743865677.230452  183376 buffer_comparator.cc:156] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-05 15:07:57.230457: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.232581  183376 buffer_comparator.cc:156] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743865677.232592  183376 buffer_comparator.cc:156] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743865677.232595  183376 buffer_comparator.cc:156] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743865677.232598  183376 buffer_comparator.cc:156] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743865677.232601  183376 buffer_comparator.cc:156] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743865677.232603  183376 buffer_comparator.cc:156] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743865677.232606  183376 buffer_comparator.cc:156] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743865677.232609  183376 buffer_comparator.cc:156] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743865677.232612  183376 buffer_comparator.cc:156] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743865677.232614  183376 buffer_comparator.cc:156] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-04-05 15:07:57.232619: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.234738  183376 buffer_comparator.cc:156] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743865677.234749  183376 buffer_comparator.cc:156] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743865677.234752  183376 buffer_comparator.cc:156] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743865677.234755  183376 buffer_comparator.cc:156] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743865677.234758  183376 buffer_comparator.cc:156] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743865677.234761  183376 buffer_comparator.cc:156] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743865677.234763  183376 buffer_comparator.cc:156] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743865677.234766  183376 buffer_comparator.cc:156] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743865677.234769  183376 buffer_comparator.cc:156] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743865677.234772  183376 buffer_comparator.cc:156] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-04-05 15:07:57.234776: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.236881  183376 buffer_comparator.cc:156] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743865677.236892  183376 buffer_comparator.cc:156] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743865677.236895  183376 buffer_comparator.cc:156] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743865677.236898  183376 buffer_comparator.cc:156] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743865677.236901  183376 buffer_comparator.cc:156] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743865677.236903  183376 buffer_comparator.cc:156] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743865677.236906  183376 buffer_comparator.cc:156] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743865677.236909  183376 buffer_comparator.cc:156] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743865677.236912  183376 buffer_comparator.cc:156] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743865677.236914  183376 buffer_comparator.cc:156] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-04-05 15:07:57.236919: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.239048  183376 buffer_comparator.cc:156] Difference at 256: nan, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743865677.239059  183376 buffer_comparator.cc:156] Difference at 257: nan, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743865677.239062  183376 buffer_comparator.cc:156] Difference at 258: nan, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743865677.239065  183376 buffer_comparator.cc:156] Difference at 259: nan, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743865677.239068  183376 buffer_comparator.cc:156] Difference at 260: nan, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743865677.239071  183376 buffer_comparator.cc:156] Difference at 261: nan, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743865677.239074  183376 buffer_comparator.cc:156] Difference at 262: nan, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743865677.239076  183376 buffer_comparator.cc:156] Difference at 263: nan, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743865677.239079  183376 buffer_comparator.cc:156] Difference at 264: nan, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743865677.239082  183376 buffer_comparator.cc:156] Difference at 265: nan, expected 15.766</span></span>
<span class="line"><span>2025-04-05 15:07:57.239086: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.241217  183376 buffer_comparator.cc:156] Difference at 256: nan, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743865677.241228  183376 buffer_comparator.cc:156] Difference at 257: nan, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743865677.241231  183376 buffer_comparator.cc:156] Difference at 258: nan, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743865677.241234  183376 buffer_comparator.cc:156] Difference at 259: nan, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743865677.241237  183376 buffer_comparator.cc:156] Difference at 260: nan, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743865677.241240  183376 buffer_comparator.cc:156] Difference at 261: nan, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743865677.241243  183376 buffer_comparator.cc:156] Difference at 262: nan, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743865677.241245  183376 buffer_comparator.cc:156] Difference at 263: nan, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743865677.241248  183376 buffer_comparator.cc:156] Difference at 264: nan, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743865677.241251  183376 buffer_comparator.cc:156] Difference at 265: nan, expected 15.766</span></span>
<span class="line"><span>2025-04-05 15:07:57.241255: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865677.243559  183376 buffer_comparator.cc:156] Difference at 256: nan, expected 11.4506</span></span>
<span class="line"><span>Epoch   1	Train Loss: 16.792463	Train Acc: 19.2857%	Val Loss: 7.485401	Val Acc: 20.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 9.168443	Train Acc: 25.0000%	Val Loss: 3.362585	Val Acc: 29.2000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.840035	Train Acc: 41.4286%	Val Loss: 2.057412	Val Acc: 37.6000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 2.189526	Train Acc: 54.2857%	Val Loss: 1.846857	Val Acc: 45.0000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.725716	Train Acc: 61.4286%	Val Loss: 1.675081	Val Acc: 49.4000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.362549	Train Acc: 67.8571%	Val Loss: 1.525527	Val Acc: 56.6000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.182950	Train Acc: 75.0000%	Val Loss: 1.450947	Val Acc: 60.6000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.088486	Train Acc: 75.7143%	Val Loss: 1.442043	Val Acc: 62.2000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 1.028590	Train Acc: 74.2857%	Val Loss: 1.465832	Val Acc: 61.4000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 0.965212	Train Acc: 75.0000%	Val Loss: 1.497198	Val Acc: 60.0000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.891447	Train Acc: 72.1429%	Val Loss: 1.610129	Val Acc: 56.6000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.870790	Train Acc: 76.4286%	Val Loss: 1.561922	Val Acc: 58.4000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.798311	Train Acc: 82.8571%	Val Loss: 1.494713	Val Acc: 62.0000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.727946	Train Acc: 84.2857%	Val Loss: 1.528784	Val Acc: 62.4000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.692915	Train Acc: 84.2857%	Val Loss: 1.576002	Val Acc: 61.2000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.676838	Train Acc: 83.5714%	Val Loss: 1.597641	Val Acc: 62.4000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.654659	Train Acc: 83.5714%	Val Loss: 1.585721	Val Acc: 62.8000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.617897	Train Acc: 85.7143%	Val Loss: 1.550064	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.574823	Train Acc: 86.4286%	Val Loss: 1.513625	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.540463	Train Acc: 86.4286%	Val Loss: 1.489742	Val Acc: 66.2000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.519199	Train Acc: 86.4286%	Val Loss: 1.476741	Val Acc: 67.2000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.502111	Train Acc: 87.1429%	Val Loss: 1.472621	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.486873	Train Acc: 87.8571%	Val Loss: 1.476589	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.472336	Train Acc: 87.1429%	Val Loss: 1.485310	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.457071	Train Acc: 89.2857%	Val Loss: 1.499061	Val Acc: 67.2000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.440181	Train Acc: 90.0000%	Val Loss: 1.519394	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.422126	Train Acc: 90.7143%	Val Loss: 1.545975	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.405241	Train Acc: 90.0000%	Val Loss: 1.580290	Val Acc: 67.0000%</span></span>
<span class="line"><span>Early Stopping at Epoch 28</span></span>
<span class="line"><span>2025-04-05 15:08:57.122333: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:08:57.148806: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:08:57.282451: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:08:57.411501: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743865737.489754  183376 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743865737.489816  183376 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743865737.489824  183376 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743865737.489832  183376 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.489839  183376 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743865737.489846  183376 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743865737.489853  183376 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.489860  183376 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743865737.489867  183376 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.489874  183376 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-05 15:08:57.489888: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.493409  183376 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743865737.493438  183376 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743865737.493446  183376 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743865737.493453  183376 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.493460  183376 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743865737.493467  183376 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743865737.493474  183376 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.493480  183376 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743865737.493487  183376 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.493494  183376 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-05 15:08:57.493506: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.497036  183376 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743865737.497048  183376 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743865737.497051  183376 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743865737.497054  183376 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.497057  183376 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743865737.497061  183376 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743865737.497066  183376 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.497069  183376 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743865737.497072  183376 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.497075  183376 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-05 15:08:57.497080: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.500475  183376 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743865737.500486  183376 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743865737.500490  183376 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743865737.500493  183376 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.500496  183376 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743865737.500499  183376 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743865737.500502  183376 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.500505  183376 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743865737.500508  183376 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.500511  183376 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-05 15:08:57.500516: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.503769  183376 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743865737.503781  183376 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743865737.503784  183376 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743865737.503788  183376 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.503791  183376 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743865737.503794  183376 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743865737.503797  183376 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.503800  183376 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743865737.503803  183376 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.503806  183376 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-05 15:08:57.503811: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.507107  183376 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743865737.507119  183376 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743865737.507122  183376 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743865737.507125  183376 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.507129  183376 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743865737.507132  183376 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743865737.507135  183376 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.507138  183376 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743865737.507143  183376 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.507146  183376 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-05 15:08:57.507151: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.510394  183376 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1743865737.510405  183376 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743865737.510409  183376 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743865737.510412  183376 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743865737.510415  183376 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743865737.510418  183376 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743865737.510421  183376 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743865737.510424  183376 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1743865737.510427  183376 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743865737.510430  183376 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-04-05 15:08:57.510435: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.513617  183376 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743865737.513630  183376 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743865737.513633  183376 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743865737.513636  183376 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743865737.513639  183376 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743865737.513643  183376 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743865737.513646  183376 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743865737.513649  183376 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1743865737.513652  183376 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1743865737.513655  183376 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-04-05 15:08:57.513660: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.516936  183376 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743865737.516948  183376 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743865737.516952  183376 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743865737.516955  183376 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743865737.516958  183376 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743865737.516961  183376 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743865737.516964  183376 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743865737.516967  183376 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.516970  183376 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743865737.516973  183376 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-05 15:08:57.516980: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.520145  183376 buffer_comparator.cc:156] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1743865737.520158  183376 buffer_comparator.cc:156] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1743865737.520163  183376 buffer_comparator.cc:156] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1743865737.520166  183376 buffer_comparator.cc:156] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1743865737.520170  183376 buffer_comparator.cc:156] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1743865737.520173  183376 buffer_comparator.cc:156] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1743865737.520177  183376 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743865737.520180  183376 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743865737.520183  183376 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743865737.520186  183376 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-04-05 15:08:57.520191: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.523451  183376 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743865737.523464  183376 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743865737.523467  183376 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743865737.523471  183376 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743865737.523474  183376 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743865737.523477  183376 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743865737.523480  183376 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743865737.523483  183376 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.523486  183376 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743865737.523489  183376 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-05 15:08:57.523494: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.526793  183376 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743865737.526806  183376 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743865737.526809  183376 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743865737.526813  183376 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743865737.526816  183376 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743865737.526819  183376 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743865737.526822  183376 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743865737.526825  183376 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743865737.526828  183376 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743865737.526831  183376 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-05 15:08:57.526836: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.530125  183376 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865737.530136  183376 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743865737.530139  183376 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743865737.530143  183376 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865737.530146  183376 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743865737.530149  183376 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.530152  183376 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.530155  183376 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743865737.530158  183376 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.530161  183376 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-05 15:08:57.530166: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.533418  183376 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865737.533429  183376 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743865737.533432  183376 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743865737.533436  183376 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865737.533439  183376 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743865737.533442  183376 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.533445  183376 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.533448  183376 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743865737.533451  183376 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.533454  183376 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-05 15:08:57.533459: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.536718  183376 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865737.536729  183376 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743865737.536732  183376 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743865737.536736  183376 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865737.536739  183376 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743865737.536742  183376 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.536745  183376 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.536748  183376 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743865737.536751  183376 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.536754  183376 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-05 15:08:57.536759: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.539962  183376 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1743865737.539975  183376 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865737.539979  183376 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743865737.539982  183376 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743865737.539985  183376 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865737.539988  183376 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743865737.539991  183376 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.539994  183376 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1743865737.539997  183376 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.540000  183376 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-04-05 15:08:57.540005: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.543207  183376 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743865737.543220  183376 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743865737.543223  183376 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743865737.543226  183376 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743865737.543229  183376 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743865737.543233  183376 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743865737.543236  183376 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.543239  183376 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743865737.543242  183376 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743865737.543245  183376 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-05 15:08:57.543249: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.546681  183376 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743865737.546693  183376 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743865737.546696  183376 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743865737.546700  183376 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.546703  183376 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743865737.546706  183376 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743865737.546709  183376 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743865737.546712  183376 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1743865737.546715  183376 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1743865737.546718  183376 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-04-05 15:08:57.546723: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865737.550175  183376 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743865737.550187  183376 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743865737.550190  183376 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743865737.550195  183376 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743865737.550198  183376 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743865737.550201  183376 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743865737.550204  183376 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743865737.550207  183376 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1743865737.550210  183376 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1743865737.550213  183376 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-04-05 15:08:57.550218: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-04-05 15:08:58.835421: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:08:59.076042: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:08:59.324133: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-05 15:08:59.764561: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743865739.772293  183376 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743865739.772342  183376 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743865739.772347  183376 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743865739.772352  183376 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.772356  183376 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743865739.772360  183376 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743865739.772364  183376 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743865739.772369  183376 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743865739.772373  183376 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.772377  183376 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-05 15:08:59.772387: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.775747  183376 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743865739.775766  183376 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743865739.775771  183376 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743865739.775775  183376 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.775779  183376 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743865739.775783  183376 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743865739.775787  183376 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743865739.775792  183376 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743865739.775796  183376 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.775802  183376 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-05 15:08:59.775809: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.779187  183376 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743865739.779205  183376 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743865739.779209  183376 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743865739.779214  183376 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.779218  183376 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743865739.779222  183376 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743865739.779226  183376 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743865739.779230  183376 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743865739.779234  183376 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.779238  183376 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-05 15:08:59.779245: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.782759  183376 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743865739.782780  183376 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743865739.782785  183376 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743865739.782789  183376 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.782794  183376 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743865739.782798  183376 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743865739.782802  183376 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743865739.782806  183376 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743865739.782810  183376 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.782814  183376 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-05 15:08:59.782821: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.786136  183376 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743865739.786149  183376 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743865739.786152  183376 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743865739.786155  183376 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.786158  183376 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743865739.786161  183376 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743865739.786164  183376 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743865739.786167  183376 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743865739.786170  183376 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.786173  183376 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-05 15:08:59.786177: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.789501  183376 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743865739.789515  183376 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743865739.789518  183376 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743865739.789521  183376 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.789524  183376 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743865739.789527  183376 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743865739.789530  183376 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743865739.789533  183376 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743865739.789536  183376 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.789539  183376 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-05 15:08:59.789544: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.792807  183376 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743865739.792820  183376 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743865739.792823  183376 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743865739.792826  183376 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743865739.792829  183376 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743865739.792832  183376 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743865739.792835  183376 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743865739.792838  183376 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743865739.792841  183376 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743865739.792844  183376 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-05 15:08:59.792848: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.796030  183376 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743865739.796042  183376 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743865739.796045  183376 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743865739.796048  183376 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743865739.796051  183376 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743865739.796054  183376 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743865739.796057  183376 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743865739.796060  183376 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743865739.796063  183376 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743865739.796066  183376 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-05 15:08:59.796071: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.799339  183376 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743865739.799351  183376 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743865739.799355  183376 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743865739.799358  183376 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743865739.799361  183376 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743865739.799365  183376 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743865739.799368  183376 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743865739.799371  183376 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743865739.799374  183376 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743865739.799377  183376 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-05 15:08:59.799382: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.802560  183376 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743865739.802573  183376 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743865739.802576  183376 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743865739.802579  183376 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743865739.802582  183376 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743865739.802585  183376 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743865739.802588  183376 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743865739.802591  183376 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743865739.802594  183376 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743865739.802597  183376 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-05 15:08:59.802602: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.805846  183376 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743865739.805859  183376 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743865739.805862  183376 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743865739.805865  183376 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743865739.805868  183376 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743865739.805871  183376 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743865739.805874  183376 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743865739.805877  183376 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743865739.805880  183376 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743865739.805883  183376 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-05 15:08:59.805887: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.809198  183376 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743865739.809211  183376 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743865739.809214  183376 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743865739.809217  183376 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743865739.809220  183376 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743865739.809223  183376 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743865739.809226  183376 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743865739.809229  183376 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743865739.809232  183376 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743865739.809237  183376 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-05 15:08:59.809241: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.812547  183376 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743865739.812560  183376 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.812563  183376 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.812566  183376 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743865739.812569  183376 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743865739.812572  183376 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743865739.812575  183376 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743865739.812578  183376 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743865739.812581  183376 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743865739.812584  183376 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-05 15:08:59.812589: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.815840  183376 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743865739.815853  183376 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.815856  183376 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.815859  183376 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743865739.815862  183376 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743865739.815865  183376 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743865739.815868  183376 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743865739.815871  183376 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743865739.815874  183376 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743865739.815877  183376 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-05 15:08:59.815882: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.819140  183376 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743865739.819153  183376 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.819156  183376 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.819159  183376 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743865739.819162  183376 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743865739.819165  183376 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743865739.819168  183376 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743865739.819171  183376 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743865739.819174  183376 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743865739.819177  183376 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-05 15:08:59.819182: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.822401  183376 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743865739.822415  183376 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.822419  183376 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.822422  183376 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743865739.822425  183376 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743865739.822428  183376 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743865739.822431  183376 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743865739.822433  183376 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743865739.822436  183376 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743865739.822439  183376 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-05 15:08:59.822444: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.825658  183376 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743865739.825670  183376 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743865739.825673  183376 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743865739.825676  183376 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743865739.825679  183376 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743865739.825682  183376 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743865739.825685  183376 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743865739.825688  183376 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743865739.825691  183376 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743865739.825693  183376 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-05 15:08:59.825698: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.829143  183376 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743865739.829157  183376 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743865739.829160  183376 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743865739.829163  183376 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743865739.829166  183376 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743865739.829169  183376 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743865739.829172  183376 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743865739.829175  183376 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743865739.829178  183376 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743865739.829181  183376 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-04-05 15:08:59.829185: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743865739.832657  183376 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743865739.832670  183376 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743865739.832673  183376 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743865739.832676  183376 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743865739.832679  183376 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743865739.832684  183376 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743865739.832687  183376 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743865739.832690  183376 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743865739.832693  183376 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743865739.832696  183376 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-04-05 15:08:59.832701: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.417417	Test Acc: 68.2000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const d=s(c,[["render",i]]);export{E as __pageData,d as default};
