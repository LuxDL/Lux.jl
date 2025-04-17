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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-04-16 03:43:54.451571: I external/xla/xla/service/service.cc:152] XLA service 0x95c99b0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-16 03:43:54.451760: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1744775034.452589   45486 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1744775034.452664   45486 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1744775034.452721   45486 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1744775034.463478   45486 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-04-16 03:45:00.108091: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 616 bytes spill stores, 616 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:00.711579: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:00.862627: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:00.976051: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:01.011771: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:01.076172: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:01.079350: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 268 bytes spill stores, 268 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:01.267924: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:01.359501: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 52 bytes spill stores, 52 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:02.147147: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:02.363791: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:02.498690: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:02.577005: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:02.787218: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:03.104041: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:03.965624: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:04.481767: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:04.923866: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 256 bytes spill stores, 256 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:45:05.288483: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1744775105.419104   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1744775105.419167   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1744775105.419175   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1744775105.419183   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1744775105.419190   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1744775105.419196   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1744775105.419203   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1744775105.419209   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1744775105.419216   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1744775105.419222   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-04-16 03:45:05.419238: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.421954   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1744775105.421981   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1744775105.421988   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1744775105.421995   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1744775105.422002   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1744775105.422008   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1744775105.422015   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1744775105.422022   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1744775105.422028   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1744775105.422035   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-04-16 03:45:05.422045: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.424459   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1744775105.424485   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1744775105.424492   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1744775105.424499   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1744775105.424506   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1744775105.424513   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1744775105.424519   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1744775105.424526   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1744775105.424533   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1744775105.424539   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-04-16 03:45:05.424549: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.426959   45486 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1744775105.426988   45486 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1744775105.426996   45486 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1744775105.427003   45486 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1744775105.427009   45486 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1744775105.427016   45486 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1744775105.427023   45486 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1744775105.427029   45486 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1744775105.427036   45486 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1744775105.427042   45486 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-04-16 03:45:05.427053: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.429198   45486 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1744775105.429211   45486 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1744775105.429214   45486 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1744775105.429217   45486 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1744775105.429220   45486 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1744775105.429223   45486 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1744775105.429226   45486 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1744775105.429229   45486 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1744775105.429232   45486 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1744775105.429235   45486 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-04-16 03:45:05.429239: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.431363   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1744775105.431376   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1744775105.431380   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1744775105.431383   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1744775105.431386   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1744775105.431389   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1744775105.431392   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1744775105.431395   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1744775105.431398   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1744775105.431400   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-04-16 03:45:05.431405: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.433559   45486 buffer_comparator.cc:156] Difference at 0: 16.5369, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1744775105.433571   45486 buffer_comparator.cc:156] Difference at 1: 19.4176, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1744775105.433574   45486 buffer_comparator.cc:156] Difference at 2: 16.204, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1744775105.433578   45486 buffer_comparator.cc:156] Difference at 6: 13.1759, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1744775105.433581   45486 buffer_comparator.cc:156] Difference at 9: 16.3002, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1744775105.433586   45486 buffer_comparator.cc:156] Difference at 11: 15.6508, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1744775105.433589   45486 buffer_comparator.cc:156] Difference at 12: 20.6885, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1744775105.433592   45486 buffer_comparator.cc:156] Difference at 13: 17.247, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1744775105.433595   45486 buffer_comparator.cc:156] Difference at 14: 14.7694, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1744775105.433598   45486 buffer_comparator.cc:156] Difference at 16: 17.2743, expected 15.1227</span></span>
<span class="line"><span>2025-04-16 03:45:05.433603: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.435737   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1744775105.435749   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1744775105.435752   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1744775105.435755   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1744775105.435758   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1744775105.435761   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1744775105.435764   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1744775105.435767   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1744775105.435770   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1744775105.435773   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-04-16 03:45:05.435778: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.437896   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1744775105.437909   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1744775105.437912   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1744775105.437915   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1744775105.437918   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1744775105.437921   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1744775105.437924   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1744775105.437927   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1744775105.437930   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1744775105.437933   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-04-16 03:45:05.437938: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.440059   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1744775105.440072   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1744775105.440075   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1744775105.440078   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1744775105.440081   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1744775105.440084   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1744775105.440087   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1744775105.440090   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1744775105.440093   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1744775105.440097   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-04-16 03:45:05.440102: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.442235   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1744775105.442248   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1744775105.442251   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1744775105.442254   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1744775105.442257   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1744775105.442260   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1744775105.442263   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1744775105.442266   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1744775105.442269   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1744775105.442272   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-04-16 03:45:05.442277: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.444401   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1744775105.444413   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1744775105.444416   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1744775105.444419   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1744775105.444422   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1744775105.444425   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1744775105.444428   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1744775105.444431   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1744775105.444434   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1744775105.444437   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-04-16 03:45:05.444442: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.446563   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1744775105.446576   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1744775105.446579   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1744775105.446582   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1744775105.446585   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1744775105.446588   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1744775105.446591   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1744775105.446594   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1744775105.446597   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1744775105.446600   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-04-16 03:45:05.446604: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.448734   45486 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1744775105.448748   45486 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1744775105.448752   45486 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1744775105.448755   45486 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1744775105.448758   45486 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1744775105.448761   45486 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1744775105.448763   45486 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1744775105.448766   45486 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1744775105.448769   45486 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1744775105.448772   45486 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-04-16 03:45:05.448777: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.450923   45486 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1744775105.450936   45486 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1744775105.450939   45486 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1744775105.450942   45486 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1744775105.450945   45486 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1744775105.450948   45486 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1744775105.450951   45486 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1744775105.450954   45486 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1744775105.450957   45486 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1744775105.450960   45486 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-04-16 03:45:05.450964: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.453095   45486 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1744775105.453108   45486 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1744775105.453111   45486 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1744775105.453114   45486 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1744775105.453117   45486 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1744775105.453121   45486 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1744775105.453125   45486 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1744775105.453128   45486 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1744775105.453132   45486 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1744775105.453135   45486 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-04-16 03:45:05.453140: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.455301   45486 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1744775105.455314   45486 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1744775105.455317   45486 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1744775105.455320   45486 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1744775105.455323   45486 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1744775105.455328   45486 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1744775105.455331   45486 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1744775105.455334   45486 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1744775105.455337   45486 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1744775105.455340   45486 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-04-16 03:45:05.455344: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.458885   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1744775105.458901   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1744775105.458904   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1744775105.458907   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1744775105.458910   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1744775105.458913   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1744775105.458916   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1744775105.458919   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1744775105.458922   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1744775105.458925   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-16 03:45:05.458930: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.461380   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1744775105.461393   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1744775105.461396   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1744775105.461399   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1744775105.461402   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1744775105.461405   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1744775105.461408   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1744775105.461411   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1744775105.461414   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1744775105.461417   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-16 03:45:05.461421: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.463835   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1744775105.463848   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1744775105.463852   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1744775105.463855   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1744775105.463858   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1744775105.463861   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1744775105.463863   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1744775105.463866   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1744775105.463869   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1744775105.463874   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-16 03:45:05.463878: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.466312   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1744775105.466326   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1744775105.466329   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1744775105.466332   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1744775105.466335   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1744775105.466338   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1744775105.466341   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1744775105.466344   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1744775105.466347   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1744775105.466350   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-04-16 03:45:05.466355: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.468773   45486 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1744775105.468786   45486 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1744775105.468789   45486 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1744775105.468792   45486 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1744775105.468795   45486 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1744775105.468798   45486 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1744775105.468801   45486 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1744775105.468804   45486 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1744775105.468807   45486 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1744775105.468810   45486 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-04-16 03:45:05.468815: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.471223   45486 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1744775105.471236   45486 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1744775105.471239   45486 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1744775105.471242   45486 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1744775105.471245   45486 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1744775105.471248   45486 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1744775105.471251   45486 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1744775105.471254   45486 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1744775105.471257   45486 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1744775105.471260   45486 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-04-16 03:45:05.471265: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.473679   45486 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1744775105.473693   45486 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1744775105.473696   45486 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1744775105.473699   45486 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1744775105.473702   45486 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1744775105.473705   45486 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1744775105.473708   45486 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1744775105.473711   45486 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1744775105.473714   45486 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1744775105.473717   45486 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-04-16 03:45:05.473722: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.476143   45486 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1744775105.476159   45486 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1744775105.476162   45486 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1744775105.476165   45486 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1744775105.476168   45486 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1744775105.476171   45486 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1744775105.476174   45486 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1744775105.476176   45486 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1744775105.476179   45486 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1744775105.476182   45486 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-04-16 03:45:05.476187: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.478635   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1744775105.478650   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1744775105.478654   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1744775105.478657   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1744775105.478660   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1744775105.478663   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1744775105.478666   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1744775105.478669   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1744775105.478672   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1744775105.478675   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-16 03:45:05.478679: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.481103   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1744775105.481116   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1744775105.481119   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1744775105.481123   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1744775105.481126   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1744775105.481131   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1744775105.481134   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1744775105.481137   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1744775105.481140   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1744775105.481142   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-16 03:45:05.481147: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.483668   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1744775105.483681   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1744775105.483684   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1744775105.483687   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1744775105.483690   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1744775105.483693   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1744775105.483696   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1744775105.483699   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1744775105.483702   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1744775105.483705   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-16 03:45:05.483710: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.486147   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1744775105.486160   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1744775105.486164   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1744775105.486167   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1744775105.486170   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1744775105.486173   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1744775105.486176   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1744775105.486178   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1744775105.486181   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1744775105.486184   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-16 03:45:05.486190: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.488614   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1744775105.488627   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1744775105.488631   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1744775105.488634   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1744775105.488637   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1744775105.488639   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1744775105.488642   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1744775105.488645   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1744775105.488648   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1744775105.488653   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-16 03:45:05.488658: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.491065   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1744775105.491079   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1744775105.491083   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1744775105.491086   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1744775105.491089   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1744775105.491092   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1744775105.491095   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1744775105.491097   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1744775105.491100   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1744775105.491103   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-04-16 03:45:05.491108: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.493551   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1744775105.493564   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1744775105.493567   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1744775105.493570   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1744775105.493573   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1744775105.493576   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1744775105.493579   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1744775105.493582   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1744775105.493585   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1744775105.493587   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-16 03:45:05.493592: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.496059   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1744775105.496073   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1744775105.496076   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1744775105.496079   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1744775105.496082   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1744775105.496085   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1744775105.496088   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1744775105.496091   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1744775105.496094   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1744775105.496097   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-16 03:45:05.496102: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.498548   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1744775105.498564   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1744775105.498567   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1744775105.498570   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1744775105.498573   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1744775105.498576   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1744775105.498579   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1744775105.498582   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1744775105.498585   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1744775105.498588   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-16 03:45:05.498593: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.501015   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1744775105.501029   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1744775105.501032   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1744775105.501035   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1744775105.501038   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1744775105.501041   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1744775105.501044   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1744775105.501046   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1744775105.501049   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1744775105.501052   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-04-16 03:45:05.501057: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.503499   45486 buffer_comparator.cc:156] Difference at 256: 0, expected 16.0393</span></span>
<span class="line"><span>E0000 00:00:1744775105.503513   45486 buffer_comparator.cc:156] Difference at 257: 0, expected 18.4933</span></span>
<span class="line"><span>E0000 00:00:1744775105.503516   45486 buffer_comparator.cc:156] Difference at 258: 0, expected 18.027</span></span>
<span class="line"><span>E0000 00:00:1744775105.503519   45486 buffer_comparator.cc:156] Difference at 259: 0, expected 20.7645</span></span>
<span class="line"><span>E0000 00:00:1744775105.503522   45486 buffer_comparator.cc:156] Difference at 260: 0, expected 18.8066</span></span>
<span class="line"><span>E0000 00:00:1744775105.503525   45486 buffer_comparator.cc:156] Difference at 261: 0, expected 17.9486</span></span>
<span class="line"><span>E0000 00:00:1744775105.503528   45486 buffer_comparator.cc:156] Difference at 262: 0, expected 16.8675</span></span>
<span class="line"><span>E0000 00:00:1744775105.503531   45486 buffer_comparator.cc:156] Difference at 263: 0, expected 18.7938</span></span>
<span class="line"><span>E0000 00:00:1744775105.503534   45486 buffer_comparator.cc:156] Difference at 264: 0, expected 16.5109</span></span>
<span class="line"><span>E0000 00:00:1744775105.503537   45486 buffer_comparator.cc:156] Difference at 265: 0, expected 20.2758</span></span>
<span class="line"><span>2025-04-16 03:45:05.503542: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.505975   45486 buffer_comparator.cc:156] Difference at 256: 0, expected 16.0393</span></span>
<span class="line"><span>E0000 00:00:1744775105.505988   45486 buffer_comparator.cc:156] Difference at 257: 0, expected 18.4933</span></span>
<span class="line"><span>E0000 00:00:1744775105.505991   45486 buffer_comparator.cc:156] Difference at 258: 0, expected 18.027</span></span>
<span class="line"><span>E0000 00:00:1744775105.505994   45486 buffer_comparator.cc:156] Difference at 259: 0, expected 20.7645</span></span>
<span class="line"><span>E0000 00:00:1744775105.505997   45486 buffer_comparator.cc:156] Difference at 260: 0, expected 18.8066</span></span>
<span class="line"><span>E0000 00:00:1744775105.506001   45486 buffer_comparator.cc:156] Difference at 261: 0, expected 17.9486</span></span>
<span class="line"><span>E0000 00:00:1744775105.506004   45486 buffer_comparator.cc:156] Difference at 262: 0, expected 16.8675</span></span>
<span class="line"><span>E0000 00:00:1744775105.506007   45486 buffer_comparator.cc:156] Difference at 263: 0, expected 18.7938</span></span>
<span class="line"><span>E0000 00:00:1744775105.506010   45486 buffer_comparator.cc:156] Difference at 264: 0, expected 16.5109</span></span>
<span class="line"><span>E0000 00:00:1744775105.506013   45486 buffer_comparator.cc:156] Difference at 265: 0, expected 20.2758</span></span>
<span class="line"><span>2025-04-16 03:45:05.506018: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.514069   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1744775105.514086   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1744775105.514090   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1744775105.514093   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1744775105.514096   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1744775105.514099   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1744775105.514102   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1744775105.514105   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1744775105.514108   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.514111   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-04-16 03:45:05.514116: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.521141   45486 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1744775105.521155   45486 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1744775105.521159   45486 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1744775105.521162   45486 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1744775105.521165   45486 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1744775105.521168   45486 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1744775105.521171   45486 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1744775105.521174   45486 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1744775105.521177   45486 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.521179   45486 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-04-16 03:45:05.521184: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.526028   45486 buffer_comparator.cc:156] Difference at 64: 0, expected 1106.21</span></span>
<span class="line"><span>E0000 00:00:1744775105.526042   45486 buffer_comparator.cc:156] Difference at 65: 0, expected 1087.83</span></span>
<span class="line"><span>E0000 00:00:1744775105.526045   45486 buffer_comparator.cc:156] Difference at 66: 0, expected 1090.54</span></span>
<span class="line"><span>E0000 00:00:1744775105.526068   45486 buffer_comparator.cc:156] Difference at 67: 0, expected 1104.23</span></span>
<span class="line"><span>E0000 00:00:1744775105.526071   45486 buffer_comparator.cc:156] Difference at 68: 0, expected 1104.3</span></span>
<span class="line"><span>E0000 00:00:1744775105.526074   45486 buffer_comparator.cc:156] Difference at 69: 0, expected 1093.45</span></span>
<span class="line"><span>E0000 00:00:1744775105.526077   45486 buffer_comparator.cc:156] Difference at 70: 0, expected 1091.52</span></span>
<span class="line"><span>E0000 00:00:1744775105.526080   45486 buffer_comparator.cc:156] Difference at 71: 0, expected 1110.4</span></span>
<span class="line"><span>E0000 00:00:1744775105.526083   45486 buffer_comparator.cc:156] Difference at 72: 0, expected 1106.92</span></span>
<span class="line"><span>E0000 00:00:1744775105.526088   45486 buffer_comparator.cc:156] Difference at 73: 0, expected 1088.44</span></span>
<span class="line"><span>2025-04-16 03:45:05.526093: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.530926   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.530939   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.530943   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.530946   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.530949   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.530952   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.530955   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.530958   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.530961   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.530964   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.530968: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.535570   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.535584   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.535587   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.535590   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.535593   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.535596   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.535599   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.535602   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.535605   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.535608   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.535613: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.540105   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.540119   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.540122   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.540125   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.540128   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.540131   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.540134   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.540137   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.540140   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.540143   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.540148: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.545095   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.545110   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.545114   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.545117   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.545120   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.545123   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.545126   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.545129   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.545132   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.545135   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.545139: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.550141   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.550154   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.550157   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.550161   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.550164   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.550167   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.550170   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.550173   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.550176   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.550179   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.550183: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.554758   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.554772   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.554776   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.554779   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.554782   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.554785   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.554788   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.554791   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.554794   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.554796   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.554801: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.559489   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.559502   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.559505   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.559508   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.559511   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.559516   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.559519   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.559522   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.559525   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.559527   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.559532: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.563887   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.563901   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.563904   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.563907   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.563910   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.563913   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.563916   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.563919   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.563922   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.563925   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.563930: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.568529   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.568552   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.568555   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.568558   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.568561   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.568564   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.568567   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.568570   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1744775105.568573   45486 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1744775105.568576   45486 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-04-16 03:45:05.568582: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775105.573156   45486 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1744775105.573172   45486 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1744775105.573176   45486 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1744775105.573179   45486 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1744775105.573182   45486 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1744775105.573185   45486 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1744775105.573188   45486 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1744775105.573191   45486 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>Epoch   1	Train Loss: 15.417609	Train Acc: 19.2857%	Val Loss: 7.626334	Val Acc: 20.4000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 9.052410	Train Acc: 23.5714%	Val Loss: 3.407886	Val Acc: 27.0000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.905093	Train Acc: 42.8571%	Val Loss: 2.063991	Val Acc: 37.6000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 2.235351	Train Acc: 51.4286%	Val Loss: 2.196240	Val Acc: 38.4000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 2.074790	Train Acc: 57.8571%	Val Loss: 1.945248	Val Acc: 44.2000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.654483	Train Acc: 65.7143%	Val Loss: 1.671895	Val Acc: 52.6000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.367258	Train Acc: 67.8571%	Val Loss: 1.552910	Val Acc: 58.2000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.206088	Train Acc: 71.4286%	Val Loss: 1.543512	Val Acc: 60.2000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 1.117982	Train Acc: 71.4286%	Val Loss: 1.579318	Val Acc: 58.6000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 1.075899	Train Acc: 72.1429%	Val Loss: 1.595436	Val Acc: 58.0000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 1.001182	Train Acc: 74.2857%	Val Loss: 1.570567	Val Acc: 58.6000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.892327	Train Acc: 75.7143%	Val Loss: 1.521401	Val Acc: 61.4000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.775864	Train Acc: 78.5714%	Val Loss: 1.487384	Val Acc: 63.4000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.717839	Train Acc: 82.1429%	Val Loss: 1.491008	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.634990	Train Acc: 83.5714%	Val Loss: 1.534554	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.567407	Train Acc: 83.5714%	Val Loss: 1.598125	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.554979	Train Acc: 85.0000%	Val Loss: 1.659654	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.551735	Train Acc: 85.0000%	Val Loss: 1.699356	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.534263	Train Acc: 84.2857%	Val Loss: 1.708084	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.498597	Train Acc: 85.0000%	Val Loss: 1.696730	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.465330	Train Acc: 85.0000%	Val Loss: 1.680628	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.445671	Train Acc: 85.0000%	Val Loss: 1.665993	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.430280	Train Acc: 86.4286%	Val Loss: 1.657657	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.418716	Train Acc: 87.8571%	Val Loss: 1.658252	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.407639	Train Acc: 87.8571%	Val Loss: 1.670418	Val Acc: 65.2000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.393848	Train Acc: 87.8571%	Val Loss: 1.690961	Val Acc: 65.2000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.381606	Train Acc: 87.1429%	Val Loss: 1.716056	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.371269	Train Acc: 87.1429%	Val Loss: 1.742408	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.361632	Train Acc: 87.8571%	Val Loss: 1.769058	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  30	Train Loss: 0.351922	Train Acc: 87.8571%	Val Loss: 1.793265	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  31	Train Loss: 0.342089	Train Acc: 90.0000%	Val Loss: 1.814874	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  32	Train Loss: 0.331745	Train Acc: 90.7143%	Val Loss: 1.834671	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  33	Train Loss: 0.321436	Train Acc: 91.4286%	Val Loss: 1.851505	Val Acc: 66.8000%</span></span>
<span class="line"><span>Early Stopping at Epoch 33</span></span>
<span class="line"><span>2025-04-16 03:46:07.740638: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:46:07.905757: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:46:08.077782: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:46:08.249498: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1744775168.256662   45486 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1744775168.256716   45486 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1744775168.256726   45486 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1744775168.256734   45486 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.256741   45486 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1744775168.256749   45486 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1744775168.256756   45486 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.256763   45486 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1744775168.256770   45486 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.256777   45486 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-16 03:46:08.256792: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.260321   45486 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1744775168.260347   45486 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1744775168.260355   45486 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1744775168.260362   45486 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.260369   45486 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1744775168.260376   45486 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1744775168.260384   45486 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.260391   45486 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1744775168.260398   45486 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.260405   45486 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-16 03:46:08.260415: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.263932   45486 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1744775168.263945   45486 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1744775168.263948   45486 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1744775168.263951   45486 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.263955   45486 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1744775168.263958   45486 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1744775168.263963   45486 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.263966   45486 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1744775168.263969   45486 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.263972   45486 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-16 03:46:08.263977: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.267390   45486 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1744775168.267403   45486 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1744775168.267406   45486 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1744775168.267409   45486 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.267412   45486 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1744775168.267415   45486 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1744775168.267419   45486 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.267422   45486 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1744775168.267425   45486 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.267428   45486 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-16 03:46:08.267433: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.270709   45486 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1744775168.270723   45486 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1744775168.270727   45486 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1744775168.270730   45486 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.270733   45486 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1744775168.270736   45486 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1744775168.270739   45486 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.270743   45486 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1744775168.270746   45486 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.270749   45486 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-16 03:46:08.270754: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.274119   45486 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1744775168.274133   45486 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1744775168.274137   45486 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1744775168.274140   45486 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.274143   45486 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1744775168.274146   45486 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1744775168.274149   45486 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.274152   45486 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1744775168.274157   45486 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.274161   45486 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-16 03:46:08.274166: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.277449   45486 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1744775168.277466   45486 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1744775168.277469   45486 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1744775168.277472   45486 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1744775168.277475   45486 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1744775168.277479   45486 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1744775168.277482   45486 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1744775168.277485   45486 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1744775168.277488   45486 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1744775168.277491   45486 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-04-16 03:46:08.277496: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.280746   45486 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1744775168.280759   45486 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1744775168.280762   45486 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1744775168.280766   45486 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1744775168.280769   45486 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1744775168.280772   45486 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1744775168.280775   45486 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1744775168.280778   45486 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1744775168.280781   45486 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1744775168.280784   45486 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-04-16 03:46:08.280789: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.284087   45486 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1744775168.284104   45486 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1744775168.284108   45486 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1744775168.284111   45486 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1744775168.284114   45486 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1744775168.284118   45486 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1744775168.284121   45486 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1744775168.284124   45486 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.284127   45486 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1744775168.284130   45486 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-16 03:46:08.284137: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.287359   45486 buffer_comparator.cc:156] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1744775168.287374   45486 buffer_comparator.cc:156] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1744775168.287378   45486 buffer_comparator.cc:156] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1744775168.287382   45486 buffer_comparator.cc:156] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1744775168.287385   45486 buffer_comparator.cc:156] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1744775168.287389   45486 buffer_comparator.cc:156] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1744775168.287392   45486 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1744775168.287395   45486 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1744775168.287398   45486 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1744775168.287401   45486 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-04-16 03:46:08.287406: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.290693   45486 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1744775168.290707   45486 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1744775168.290711   45486 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1744775168.290714   45486 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1744775168.290717   45486 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1744775168.290720   45486 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1744775168.290723   45486 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1744775168.290726   45486 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.290729   45486 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1744775168.290732   45486 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-16 03:46:08.290738: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.294117   45486 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1744775168.294133   45486 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1744775168.294137   45486 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1744775168.294140   45486 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1744775168.294143   45486 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1744775168.294146   45486 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1744775168.294149   45486 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1744775168.294152   45486 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1744775168.294155   45486 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1744775168.294158   45486 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-16 03:46:08.294164: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.297521   45486 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1744775168.297536   45486 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1744775168.297540   45486 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1744775168.297543   45486 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1744775168.297546   45486 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1744775168.297550   45486 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.297553   45486 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.297556   45486 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1744775168.297559   45486 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.297562   45486 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-16 03:46:08.297568: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.300888   45486 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1744775168.300904   45486 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1744775168.300908   45486 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1744775168.300911   45486 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1744775168.300914   45486 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1744775168.300917   45486 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.300921   45486 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.300924   45486 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1744775168.300927   45486 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.300930   45486 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-16 03:46:08.300935: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.304266   45486 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1744775168.304284   45486 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1744775168.304287   45486 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1744775168.304291   45486 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1744775168.304294   45486 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1744775168.304297   45486 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.304300   45486 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.304303   45486 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1744775168.304306   45486 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.304309   45486 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-16 03:46:08.304315: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.307586   45486 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1744775168.307606   45486 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1744775168.307610   45486 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1744775168.307613   45486 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1744775168.307616   45486 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1744775168.307620   45486 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1744775168.307623   45486 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.307626   45486 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1744775168.307629   45486 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.307632   45486 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-04-16 03:46:08.307638: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.310917   45486 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1744775168.310931   45486 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1744775168.310935   45486 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1744775168.310939   45486 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1744775168.310942   45486 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1744775168.310945   45486 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1744775168.310948   45486 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.310951   45486 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1744775168.310954   45486 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1744775168.310957   45486 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-16 03:46:08.310962: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.314569   45486 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1744775168.314585   45486 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1744775168.314588   45486 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1744775168.314591   45486 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.314595   45486 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1744775168.314598   45486 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1744775168.314601   45486 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1744775168.314604   45486 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1744775168.314607   45486 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1744775168.314610   45486 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-04-16 03:46:08.314615: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775168.318207   45486 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1744775168.318221   45486 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1744775168.318224   45486 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1744775168.318229   45486 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1744775168.318233   45486 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1744775168.318236   45486 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1744775168.318239   45486 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1744775168.318242   45486 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1744775168.318245   45486 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1744775168.318248   45486 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-04-16 03:46:08.318253: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-04-16 03:46:09.813183: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:46:09.958854: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:46:10.167494: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-16 03:46:10.353075: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1744775170.558874   45486 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1744775170.558916   45486 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1744775170.558920   45486 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1744775170.558923   45486 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.558927   45486 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1744775170.558930   45486 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1744775170.558933   45486 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1744775170.558936   45486 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1744775170.558939   45486 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.558942   45486 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-16 03:46:10.558951: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.561422   45486 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1744775170.561433   45486 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1744775170.561437   45486 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1744775170.561440   45486 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.561443   45486 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1744775170.561446   45486 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1744775170.561449   45486 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1744775170.561452   45486 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1744775170.561455   45486 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.561460   45486 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-16 03:46:10.561465: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.564199   45486 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1744775170.564211   45486 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1744775170.564214   45486 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1744775170.564217   45486 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.564220   45486 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1744775170.564223   45486 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1744775170.564226   45486 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1744775170.564229   45486 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1744775170.564232   45486 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.564235   45486 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-16 03:46:10.564240: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.567617   45486 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1744775170.567629   45486 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1744775170.567633   45486 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1744775170.567636   45486 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.567639   45486 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1744775170.567642   45486 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1744775170.567645   45486 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1744775170.567648   45486 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1744775170.567651   45486 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.567653   45486 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-16 03:46:10.567658: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.570535   45486 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1744775170.570547   45486 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1744775170.570551   45486 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1744775170.570554   45486 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.570557   45486 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1744775170.570560   45486 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1744775170.570563   45486 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1744775170.570566   45486 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1744775170.570569   45486 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.570572   45486 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-16 03:46:10.570576: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.573389   45486 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1744775170.573402   45486 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1744775170.573406   45486 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1744775170.573409   45486 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.573412   45486 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1744775170.573415   45486 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1744775170.573418   45486 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1744775170.573421   45486 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1744775170.573424   45486 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.573427   45486 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-16 03:46:10.573431: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.575997   45486 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1744775170.576009   45486 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1744775170.576013   45486 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1744775170.576016   45486 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1744775170.576019   45486 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1744775170.576022   45486 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1744775170.576025   45486 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1744775170.576027   45486 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1744775170.576030   45486 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1744775170.576033   45486 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-16 03:46:10.576038: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.578879   45486 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1744775170.578890   45486 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1744775170.578894   45486 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1744775170.578897   45486 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1744775170.578900   45486 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1744775170.578903   45486 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1744775170.578906   45486 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1744775170.578909   45486 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1744775170.578912   45486 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1744775170.578915   45486 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-16 03:46:10.578919: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.582301   45486 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1744775170.582313   45486 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1744775170.582316   45486 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1744775170.582319   45486 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1744775170.582322   45486 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1744775170.582327   45486 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1744775170.582330   45486 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1744775170.582333   45486 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1744775170.582336   45486 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1744775170.582339   45486 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-16 03:46:10.582344: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.585272   45486 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1744775170.585284   45486 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1744775170.585287   45486 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1744775170.585290   45486 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1744775170.585293   45486 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1744775170.585296   45486 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1744775170.585299   45486 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1744775170.585302   45486 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1744775170.585305   45486 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1744775170.585308   45486 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-16 03:46:10.585313: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.587904   45486 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1744775170.587917   45486 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1744775170.587920   45486 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1744775170.587924   45486 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1744775170.587927   45486 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1744775170.587930   45486 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1744775170.587933   45486 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1744775170.587935   45486 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1744775170.587938   45486 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1744775170.587941   45486 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-16 03:46:10.587946: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.591224   45486 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1744775170.591236   45486 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1744775170.591239   45486 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1744775170.591242   45486 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1744775170.591245   45486 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1744775170.591248   45486 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1744775170.591251   45486 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1744775170.591254   45486 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1744775170.591257   45486 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1744775170.591262   45486 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-16 03:46:10.591267: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.594233   45486 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1744775170.594245   45486 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.594248   45486 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.594251   45486 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1744775170.594254   45486 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1744775170.594257   45486 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1744775170.594260   45486 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1744775170.594263   45486 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1744775170.594266   45486 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1744775170.594269   45486 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-16 03:46:10.594274: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.597872   45486 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1744775170.597930   45486 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.597939   45486 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.597946   45486 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1744775170.597953   45486 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1744775170.597960   45486 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1744775170.597967   45486 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1744775170.597974   45486 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1744775170.597980   45486 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1744775170.597987   45486 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-16 03:46:10.598002: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.601873   45486 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1744775170.601906   45486 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.601913   45486 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.601920   45486 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1744775170.601927   45486 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1744775170.601934   45486 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1744775170.601941   45486 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1744775170.601947   45486 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1744775170.601954   45486 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1744775170.601961   45486 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-16 03:46:10.601972: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.605316   45486 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1744775170.605345   45486 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.605352   45486 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.605359   45486 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1744775170.605366   45486 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1744775170.605373   45486 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1744775170.605379   45486 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1744775170.605386   45486 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1744775170.605393   45486 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1744775170.605399   45486 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-16 03:46:10.605410: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.608636   45486 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1744775170.608649   45486 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1744775170.608652   45486 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1744775170.608655   45486 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1744775170.608658   45486 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1744775170.608661   45486 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1744775170.608664   45486 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1744775170.608667   45486 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1744775170.608670   45486 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1744775170.608673   45486 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-16 03:46:10.608678: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.611465   45486 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1744775170.611479   45486 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1744775170.611483   45486 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1744775170.611487   45486 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1744775170.611491   45486 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1744775170.611494   45486 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1744775170.611497   45486 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1744775170.611500   45486 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1744775170.611503   45486 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1744775170.611506   45486 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-04-16 03:46:10.611511: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1744775170.614940   45486 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1744775170.614953   45486 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1744775170.614956   45486 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1744775170.614959   45486 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1744775170.614962   45486 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1744775170.614967   45486 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1744775170.614970   45486 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1744775170.614974   45486 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1744775170.614977   45486 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1744775170.614980   45486 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-04-16 03:46:10.614984: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.630360	Test Acc: 68.9000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const d=s(c,[["render",i]]);export{E as __pageData,d as default};
