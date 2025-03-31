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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-03-31 06:38:23.995301: I external/xla/xla/service/service.cc:152] XLA service 0x8946390 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-31 06:38:23.995653: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743403103.997285 3426407 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743403103.997668 3426407 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743403103.997996 3426407 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743403104.014934 3426407 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-7/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-7/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-03-31 06:39:21.447436: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:21.500153: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:21.706802: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:21.763017: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:21.995289: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:22.192864: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:22.993966: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:23.146728: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 616 bytes spill stores, 616 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:23.223675: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 268 bytes spill stores, 268 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:23.354474: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:24.553372: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 52 bytes spill stores, 52 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:24.878759: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:25.408057: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:25.433404: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:25.621571: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:25.666879: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:25.687751: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:25.741100: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:39:25.750014: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 256 bytes spill stores, 256 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743403165.939021 3426407 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.940009 3426407 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743403165.940018 3426407 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743403165.940025 3426407 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743403165.940032 3426407 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743403165.940039 3426407 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743403165.940045 3426407 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403165.940052 3426407 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743403165.940059 3426407 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743403165.940066 3426407 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-31 06:39:25.940081: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.943882 3426407 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.943911 3426407 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743403165.943918 3426407 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743403165.943925 3426407 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743403165.943932 3426407 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743403165.943939 3426407 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743403165.943945 3426407 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403165.943952 3426407 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743403165.943959 3426407 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743403165.943965 3426407 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-31 06:39:25.943976: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.947378 3426407 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.947394 3426407 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743403165.947398 3426407 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743403165.947401 3426407 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743403165.947404 3426407 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743403165.947407 3426407 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743403165.947410 3426407 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403165.947413 3426407 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743403165.947416 3426407 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743403165.947419 3426407 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-31 06:39:25.947425: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.950663 3426407 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.950676 3426407 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743403165.950679 3426407 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743403165.950682 3426407 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743403165.950685 3426407 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743403165.950688 3426407 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743403165.950691 3426407 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403165.950694 3426407 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743403165.950697 3426407 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743403165.950700 3426407 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-31 06:39:25.950705: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.953876 3426407 buffer_comparator.cc:156] Difference at 0: 1140.51, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743403165.953890 3426407 buffer_comparator.cc:156] Difference at 1: 1405.76, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743403165.953893 3426407 buffer_comparator.cc:156] Difference at 2: 2133.59, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743403165.953896 3426407 buffer_comparator.cc:156] Difference at 3: 1840.1, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743403165.953899 3426407 buffer_comparator.cc:156] Difference at 4: 1308.3, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743403165.953902 3426407 buffer_comparator.cc:156] Difference at 5: 2065.73, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.953905 3426407 buffer_comparator.cc:156] Difference at 6: 1481.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1743403165.953908 3426407 buffer_comparator.cc:156] Difference at 7: 1113.93, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743403165.953911 3426407 buffer_comparator.cc:156] Difference at 8: 1359.57, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743403165.953914 3426407 buffer_comparator.cc:156] Difference at 9: 2049.53, expected 1833.76</span></span>
<span class="line"><span>2025-03-31 06:39:25.953919: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.957117 3426407 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.957131 3426407 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743403165.957134 3426407 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743403165.957137 3426407 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743403165.957140 3426407 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743403165.957143 3426407 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743403165.957146 3426407 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403165.957149 3426407 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743403165.957152 3426407 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743403165.957155 3426407 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-31 06:39:25.957160: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.960345 3426407 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743403165.960361 3426407 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743403165.960364 3426407 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743403165.960368 3426407 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743403165.960371 3426407 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743403165.960374 3426407 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743403165.960377 3426407 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743403165.960380 3426407 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743403165.960383 3426407 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743403165.960386 3426407 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-31 06:39:25.960390: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.963504 3426407 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743403165.963517 3426407 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743403165.963521 3426407 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743403165.963524 3426407 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743403165.963527 3426407 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743403165.963530 3426407 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743403165.963533 3426407 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743403165.963536 3426407 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743403165.963539 3426407 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743403165.963542 3426407 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-31 06:39:25.963546: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.966716 3426407 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743403165.966732 3426407 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743403165.966735 3426407 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743403165.966738 3426407 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743403165.966741 3426407 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.966744 3426407 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.966747 3426407 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743403165.966750 3426407 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743403165.966753 3426407 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743403165.966756 3426407 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-31 06:39:25.966760: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.969869 3426407 buffer_comparator.cc:156] Difference at 0: 1057.99, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743403165.969888 3426407 buffer_comparator.cc:156] Difference at 1: 1320.07, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743403165.969892 3426407 buffer_comparator.cc:156] Difference at 2: 2005.81, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743403165.969895 3426407 buffer_comparator.cc:156] Difference at 3: 1746.91, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743403165.969898 3426407 buffer_comparator.cc:156] Difference at 4: 1253.09, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743403165.969901 3426407 buffer_comparator.cc:156] Difference at 7: 1176.41, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743403165.969904 3426407 buffer_comparator.cc:156] Difference at 8: 1399.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743403165.969907 3426407 buffer_comparator.cc:156] Difference at 9: 2127.08, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1743403165.969910 3426407 buffer_comparator.cc:156] Difference at 10: 1879.74, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1743403165.969913 3426407 buffer_comparator.cc:156] Difference at 11: 1363.62, expected 1121.95</span></span>
<span class="line"><span>2025-03-31 06:39:25.969917: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.973091 3426407 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743403165.973107 3426407 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743403165.973111 3426407 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743403165.973114 3426407 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743403165.973117 3426407 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.973120 3426407 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.973123 3426407 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743403165.973126 3426407 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743403165.973129 3426407 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743403165.973132 3426407 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-31 06:39:25.973136: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.976294 3426407 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743403165.976307 3426407 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743403165.976311 3426407 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743403165.976314 3426407 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743403165.976317 3426407 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.976320 3426407 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.976323 3426407 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743403165.976326 3426407 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743403165.976328 3426407 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743403165.976331 3426407 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-31 06:39:25.976336: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.979613 3426407 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743403165.979628 3426407 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403165.979631 3426407 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.979636 3426407 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743403165.979639 3426407 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403165.979642 3426407 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.979645 3426407 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.979648 3426407 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743403165.979651 3426407 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743403165.979654 3426407 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-31 06:39:25.979659: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.982917 3426407 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743403165.982936 3426407 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403165.982939 3426407 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.982942 3426407 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743403165.982945 3426407 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403165.982948 3426407 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.982951 3426407 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.982954 3426407 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743403165.982957 3426407 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743403165.982960 3426407 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-31 06:39:25.982965: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.986240 3426407 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743403165.986255 3426407 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403165.986258 3426407 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.986261 3426407 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743403165.986264 3426407 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403165.986267 3426407 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.986270 3426407 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.986273 3426407 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743403165.986276 3426407 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743403165.986279 3426407 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-31 06:39:25.986284: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.989491 3426407 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743403165.989508 3426407 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403165.989511 3426407 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.989514 3426407 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743403165.989517 3426407 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403165.989522 3426407 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.989525 3426407 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.989528 3426407 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743403165.989531 3426407 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743403165.989534 3426407 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-31 06:39:25.989538: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.992675 3426407 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743403165.992690 3426407 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403165.992693 3426407 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743403165.992696 3426407 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743403165.992699 3426407 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403165.992702 3426407 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.992705 3426407 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403165.992707 3426407 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743403165.992710 3426407 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743403165.992713 3426407 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-31 06:39:25.992718: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.996161 3426407 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743403165.996175 3426407 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743403165.996178 3426407 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.996181 3426407 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743403165.996184 3426407 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743403165.996187 3426407 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743403165.996190 3426407 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743403165.996193 3426407 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743403165.996196 3426407 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743403165.996199 3426407 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-31 06:39:25.996204: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403165.999669 3426407 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743403165.999687 3426407 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743403165.999690 3426407 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743403165.999693 3426407 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743403165.999696 3426407 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743403165.999699 3426407 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743403165.999702 3426407 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743403165.999707 3426407 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743403165.999710 3426407 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743403165.999713 3426407 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-31 06:39:25.999718: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.006227 3426407 buffer_comparator.cc:156] Difference at 16: -nan, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1743403166.006251 3426407 buffer_comparator.cc:156] Difference at 17: -nan, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1743403166.006254 3426407 buffer_comparator.cc:156] Difference at 18: -nan, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1743403166.006257 3426407 buffer_comparator.cc:156] Difference at 19: -nan, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1743403166.006260 3426407 buffer_comparator.cc:156] Difference at 20: -nan, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1743403166.006262 3426407 buffer_comparator.cc:156] Difference at 21: -nan, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1743403166.006265 3426407 buffer_comparator.cc:156] Difference at 22: -nan, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1743403166.006268 3426407 buffer_comparator.cc:156] Difference at 23: -nan, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1743403166.006271 3426407 buffer_comparator.cc:156] Difference at 24: -nan, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1743403166.006273 3426407 buffer_comparator.cc:156] Difference at 25: -nan, expected 357.082</span></span>
<span class="line"><span>2025-03-31 06:39:26.006279: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.010995 3426407 buffer_comparator.cc:156] Difference at 16: -nan, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1743403166.011011 3426407 buffer_comparator.cc:156] Difference at 17: -nan, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1743403166.011014 3426407 buffer_comparator.cc:156] Difference at 18: -nan, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1743403166.011016 3426407 buffer_comparator.cc:156] Difference at 19: -nan, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1743403166.011019 3426407 buffer_comparator.cc:156] Difference at 20: -nan, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1743403166.011022 3426407 buffer_comparator.cc:156] Difference at 21: -nan, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1743403166.011025 3426407 buffer_comparator.cc:156] Difference at 22: -nan, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1743403166.011027 3426407 buffer_comparator.cc:156] Difference at 23: -nan, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1743403166.011030 3426407 buffer_comparator.cc:156] Difference at 24: -nan, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1743403166.011033 3426407 buffer_comparator.cc:156] Difference at 25: -nan, expected 357.082</span></span>
<span class="line"><span>2025-03-31 06:39:26.011038: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.014434 3426407 buffer_comparator.cc:156] Difference at 64: -nan, expected 357.295</span></span>
<span class="line"><span>E0000 00:00:1743403166.014450 3426407 buffer_comparator.cc:156] Difference at 65: -nan, expected 365.079</span></span>
<span class="line"><span>E0000 00:00:1743403166.014453 3426407 buffer_comparator.cc:156] Difference at 66: -nan, expected 364.297</span></span>
<span class="line"><span>E0000 00:00:1743403166.014456 3426407 buffer_comparator.cc:156] Difference at 67: -nan, expected 356.584</span></span>
<span class="line"><span>E0000 00:00:1743403166.014458 3426407 buffer_comparator.cc:156] Difference at 68: -nan, expected 350.44</span></span>
<span class="line"><span>E0000 00:00:1743403166.014461 3426407 buffer_comparator.cc:156] Difference at 69: -nan, expected 355.742</span></span>
<span class="line"><span>E0000 00:00:1743403166.014464 3426407 buffer_comparator.cc:156] Difference at 70: -nan, expected 347.459</span></span>
<span class="line"><span>E0000 00:00:1743403166.014467 3426407 buffer_comparator.cc:156] Difference at 71: -nan, expected 364.613</span></span>
<span class="line"><span>E0000 00:00:1743403166.014469 3426407 buffer_comparator.cc:156] Difference at 72: -nan, expected 362.734</span></span>
<span class="line"><span>E0000 00:00:1743403166.014472 3426407 buffer_comparator.cc:156] Difference at 73: -nan, expected 362.087</span></span>
<span class="line"><span>2025-03-31 06:39:26.014478: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.017790 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.017804 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.017807 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.017810 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.017813 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.017815 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.017818 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.017821 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.017824 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.017826 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.017831: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.021127 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.021144 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.021147 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.021150 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.021152 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.021155 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.021158 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.021161 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.021163 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.021166 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.021171: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.024363 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.024378 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.024381 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.024384 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.024387 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.024390 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.024393 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.024395 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.024398 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.024401 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.024405: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.027735 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.027751 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.027754 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.027757 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.027759 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.027762 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.027765 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.027768 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.027770 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.027773 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.027778: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.030896 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.030912 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.030915 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.030918 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.030921 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.030924 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.030926 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.030929 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.030932 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.030934 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.030939: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.034299 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.034315 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.034317 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.034320 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.034323 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.034326 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.034329 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.034331 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.034334 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.034337 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.034341: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.037675 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.037689 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.037692 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.037695 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.037699 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.037702 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.037705 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.037708 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.037710 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.037713 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.037718: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.040880 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.040898 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.040901 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.040904 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.040906 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.040909 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.040912 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.040914 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.040917 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.040920 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.040924: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.044274 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.044287 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.044291 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.044293 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.044296 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.044299 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.044302 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.044304 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.044307 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.044310 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.044314: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.047429 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.047442 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.047445 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.047448 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.047451 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.047454 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.047456 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.047460 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.047463 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.047466 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.047470: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.050614 3426407 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.050628 3426407 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.050631 3426407 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.050634 3426407 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.050637 3426407 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.050640 3426407 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.050642 3426407 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.050645 3426407 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.050648 3426407 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.050651 3426407 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.050655: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.053836 3426407 buffer_comparator.cc:156] Difference at 128: nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743403166.053850 3426407 buffer_comparator.cc:156] Difference at 129: nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743403166.053853 3426407 buffer_comparator.cc:156] Difference at 130: nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743403166.053855 3426407 buffer_comparator.cc:156] Difference at 131: nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743403166.053858 3426407 buffer_comparator.cc:156] Difference at 132: nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743403166.053861 3426407 buffer_comparator.cc:156] Difference at 133: nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743403166.053864 3426407 buffer_comparator.cc:156] Difference at 134: nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743403166.053866 3426407 buffer_comparator.cc:156] Difference at 135: nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743403166.053869 3426407 buffer_comparator.cc:156] Difference at 136: nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743403166.053872 3426407 buffer_comparator.cc:156] Difference at 137: nan, expected 357.638</span></span>
<span class="line"><span>2025-03-31 06:39:26.053876: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.056982 3426407 buffer_comparator.cc:156] Difference at 256: nan, expected 359.809</span></span>
<span class="line"><span>E0000 00:00:1743403166.056995 3426407 buffer_comparator.cc:156] Difference at 257: nan, expected 357.176</span></span>
<span class="line"><span>E0000 00:00:1743403166.056998 3426407 buffer_comparator.cc:156] Difference at 258: nan, expected 348.258</span></span>
<span class="line"><span>E0000 00:00:1743403166.057001 3426407 buffer_comparator.cc:156] Difference at 259: nan, expected 361.414</span></span>
<span class="line"><span>E0000 00:00:1743403166.057003 3426407 buffer_comparator.cc:156] Difference at 260: nan, expected 354.785</span></span>
<span class="line"><span>E0000 00:00:1743403166.057006 3426407 buffer_comparator.cc:156] Difference at 261: nan, expected 363.226</span></span>
<span class="line"><span>E0000 00:00:1743403166.057009 3426407 buffer_comparator.cc:156] Difference at 262: nan, expected 365.15</span></span>
<span class="line"><span>E0000 00:00:1743403166.057012 3426407 buffer_comparator.cc:156] Difference at 263: nan, expected 370.48</span></span>
<span class="line"><span>E0000 00:00:1743403166.057014 3426407 buffer_comparator.cc:156] Difference at 264: nan, expected 348.691</span></span>
<span class="line"><span>E0000 00:00:1743403166.057017 3426407 buffer_comparator.cc:156] Difference at 265: nan, expected 350.299</span></span>
<span class="line"><span>2025-03-31 06:39:26.057023: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.060805 3426407 buffer_comparator.cc:156] Difference at 16: 0.494975, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1743403166.060824 3426407 buffer_comparator.cc:156] Difference at 17: 0.429921, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1743403166.060827 3426407 buffer_comparator.cc:156] Difference at 18: 0.78598, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1743403166.060830 3426407 buffer_comparator.cc:156] Difference at 19: 0.887771, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1743403166.060833 3426407 buffer_comparator.cc:156] Difference at 20: 0.46468, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1743403166.060836 3426407 buffer_comparator.cc:156] Difference at 21: 0.391183, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1743403166.060839 3426407 buffer_comparator.cc:156] Difference at 22: 0.883032, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1743403166.060842 3426407 buffer_comparator.cc:156] Difference at 23: 0.769114, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1743403166.060845 3426407 buffer_comparator.cc:156] Difference at 24: 0.647369, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1743403166.060848 3426407 buffer_comparator.cc:156] Difference at 25: 0.396792, expected 18.5767</span></span>
<span class="line"><span>2025-03-31 06:39:26.060854: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.063379 3426407 buffer_comparator.cc:156] Difference at 16: 0.494975, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1743403166.063399 3426407 buffer_comparator.cc:156] Difference at 17: 0.429921, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1743403166.063402 3426407 buffer_comparator.cc:156] Difference at 18: 0.78598, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1743403166.063405 3426407 buffer_comparator.cc:156] Difference at 19: 0.887771, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1743403166.063408 3426407 buffer_comparator.cc:156] Difference at 20: 0.46468, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1743403166.063411 3426407 buffer_comparator.cc:156] Difference at 21: 0.391183, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1743403166.063414 3426407 buffer_comparator.cc:156] Difference at 22: 0.883032, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1743403166.063417 3426407 buffer_comparator.cc:156] Difference at 23: 0.769114, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1743403166.063420 3426407 buffer_comparator.cc:156] Difference at 24: 0.647369, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1743403166.063422 3426407 buffer_comparator.cc:156] Difference at 25: 0.396792, expected 18.5767</span></span>
<span class="line"><span>2025-03-31 06:39:26.063427: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.065902 3426407 buffer_comparator.cc:156] Difference at 16: 0.494975, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1743403166.065916 3426407 buffer_comparator.cc:156] Difference at 17: 0.429921, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1743403166.065919 3426407 buffer_comparator.cc:156] Difference at 18: 0.78598, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1743403166.065922 3426407 buffer_comparator.cc:156] Difference at 19: 0.887771, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1743403166.065925 3426407 buffer_comparator.cc:156] Difference at 20: 0.46468, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1743403166.065928 3426407 buffer_comparator.cc:156] Difference at 21: 0.391183, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1743403166.065931 3426407 buffer_comparator.cc:156] Difference at 22: 0.883032, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1743403166.065934 3426407 buffer_comparator.cc:156] Difference at 23: 0.769114, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1743403166.065937 3426407 buffer_comparator.cc:156] Difference at 24: 0.647369, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1743403166.065939 3426407 buffer_comparator.cc:156] Difference at 25: 0.396792, expected 18.5767</span></span>
<span class="line"><span>2025-03-31 06:39:26.065944: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.068421 3426407 buffer_comparator.cc:156] Difference at 16: 0.494975, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1743403166.068435 3426407 buffer_comparator.cc:156] Difference at 17: 0.429921, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1743403166.068438 3426407 buffer_comparator.cc:156] Difference at 18: 0.78598, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1743403166.068441 3426407 buffer_comparator.cc:156] Difference at 19: 0.887771, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1743403166.068444 3426407 buffer_comparator.cc:156] Difference at 20: 0.46468, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1743403166.068447 3426407 buffer_comparator.cc:156] Difference at 21: 0.391183, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1743403166.068450 3426407 buffer_comparator.cc:156] Difference at 22: 0.883032, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1743403166.068453 3426407 buffer_comparator.cc:156] Difference at 23: 0.769114, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1743403166.068456 3426407 buffer_comparator.cc:156] Difference at 24: 0.647369, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1743403166.068459 3426407 buffer_comparator.cc:156] Difference at 25: 0.396792, expected 18.5767</span></span>
<span class="line"><span>2025-03-31 06:39:26.068463: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.070948 3426407 buffer_comparator.cc:156] Difference at 32: 0.58509, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1743403166.070962 3426407 buffer_comparator.cc:156] Difference at 33: 0.216019, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1743403166.070965 3426407 buffer_comparator.cc:156] Difference at 34: 0.105361, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1743403166.070968 3426407 buffer_comparator.cc:156] Difference at 35: 0.790407, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1743403166.070971 3426407 buffer_comparator.cc:156] Difference at 36: 0.99032, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1743403166.070974 3426407 buffer_comparator.cc:156] Difference at 37: 0.949304, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1743403166.070977 3426407 buffer_comparator.cc:156] Difference at 38: 0.229992, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1743403166.070980 3426407 buffer_comparator.cc:156] Difference at 39: 0.327565, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1743403166.070983 3426407 buffer_comparator.cc:156] Difference at 40: 0.332908, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1743403166.070986 3426407 buffer_comparator.cc:156] Difference at 41: 0.671264, expected 20.3484</span></span>
<span class="line"><span>2025-03-31 06:39:26.070990: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.073470 3426407 buffer_comparator.cc:156] Difference at 32: 0.58509, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1743403166.073485 3426407 buffer_comparator.cc:156] Difference at 33: 0.216019, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1743403166.073488 3426407 buffer_comparator.cc:156] Difference at 34: 0.105361, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1743403166.073491 3426407 buffer_comparator.cc:156] Difference at 35: 0.790407, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1743403166.073494 3426407 buffer_comparator.cc:156] Difference at 36: 0.99032, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1743403166.073497 3426407 buffer_comparator.cc:156] Difference at 37: 0.949304, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1743403166.073499 3426407 buffer_comparator.cc:156] Difference at 38: 0.229992, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1743403166.073502 3426407 buffer_comparator.cc:156] Difference at 39: 0.327565, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1743403166.073505 3426407 buffer_comparator.cc:156] Difference at 40: 0.332908, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1743403166.073508 3426407 buffer_comparator.cc:156] Difference at 41: 0.671264, expected 20.3484</span></span>
<span class="line"><span>2025-03-31 06:39:26.073512: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.076002 3426407 buffer_comparator.cc:156] Difference at 32: 0.58509, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1743403166.076017 3426407 buffer_comparator.cc:156] Difference at 33: 0.216019, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1743403166.076020 3426407 buffer_comparator.cc:156] Difference at 34: 0.105361, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1743403166.076023 3426407 buffer_comparator.cc:156] Difference at 35: 0.790407, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1743403166.076026 3426407 buffer_comparator.cc:156] Difference at 36: 0.99032, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1743403166.076029 3426407 buffer_comparator.cc:156] Difference at 37: 0.949304, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1743403166.076032 3426407 buffer_comparator.cc:156] Difference at 38: 0.229992, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1743403166.076035 3426407 buffer_comparator.cc:156] Difference at 39: 0.327565, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1743403166.076038 3426407 buffer_comparator.cc:156] Difference at 40: 0.332908, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1743403166.076041 3426407 buffer_comparator.cc:156] Difference at 41: 0.671264, expected 20.3484</span></span>
<span class="line"><span>2025-03-31 06:39:26.076045: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.078513 3426407 buffer_comparator.cc:156] Difference at 32: 0.58509, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1743403166.078528 3426407 buffer_comparator.cc:156] Difference at 33: 0.216019, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1743403166.078531 3426407 buffer_comparator.cc:156] Difference at 34: 0.105361, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1743403166.078534 3426407 buffer_comparator.cc:156] Difference at 35: 0.790407, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1743403166.078537 3426407 buffer_comparator.cc:156] Difference at 36: 0.99032, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1743403166.078540 3426407 buffer_comparator.cc:156] Difference at 37: 0.949304, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1743403166.078543 3426407 buffer_comparator.cc:156] Difference at 38: 0.229992, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1743403166.078546 3426407 buffer_comparator.cc:156] Difference at 39: 0.327565, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1743403166.078549 3426407 buffer_comparator.cc:156] Difference at 40: 0.332908, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1743403166.078551 3426407 buffer_comparator.cc:156] Difference at 41: 0.671264, expected 20.3484</span></span>
<span class="line"><span>2025-03-31 06:39:26.078556: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.081016 3426407 buffer_comparator.cc:156] Difference at 64: 0.995594, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1743403166.081030 3426407 buffer_comparator.cc:156] Difference at 65: 0.406955, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1743403166.081033 3426407 buffer_comparator.cc:156] Difference at 66: 0.639394, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1743403166.081036 3426407 buffer_comparator.cc:156] Difference at 67: 0.748706, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1743403166.081039 3426407 buffer_comparator.cc:156] Difference at 68: 0.439518, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1743403166.081042 3426407 buffer_comparator.cc:156] Difference at 69: 0.825584, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1743403166.081045 3426407 buffer_comparator.cc:156] Difference at 70: 0.0283883, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1743403166.081048 3426407 buffer_comparator.cc:156] Difference at 71: 0.789963, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1743403166.081051 3426407 buffer_comparator.cc:156] Difference at 72: 0.0831509, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1743403166.081054 3426407 buffer_comparator.cc:156] Difference at 73: 0.318524, expected 17.8359</span></span>
<span class="line"><span>2025-03-31 06:39:26.081058: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.083524 3426407 buffer_comparator.cc:156] Difference at 64: 0.995594, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1743403166.083542 3426407 buffer_comparator.cc:156] Difference at 65: 0.406955, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1743403166.083545 3426407 buffer_comparator.cc:156] Difference at 66: 0.639394, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1743403166.083550 3426407 buffer_comparator.cc:156] Difference at 67: 0.748706, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1743403166.083553 3426407 buffer_comparator.cc:156] Difference at 68: 0.439518, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1743403166.083556 3426407 buffer_comparator.cc:156] Difference at 69: 0.825584, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1743403166.083559 3426407 buffer_comparator.cc:156] Difference at 70: 0.0283883, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1743403166.083562 3426407 buffer_comparator.cc:156] Difference at 71: 0.789963, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1743403166.083565 3426407 buffer_comparator.cc:156] Difference at 72: 0.0831509, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1743403166.083568 3426407 buffer_comparator.cc:156] Difference at 73: 0.318524, expected 17.8359</span></span>
<span class="line"><span>2025-03-31 06:39:26.083573: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.086038 3426407 buffer_comparator.cc:156] Difference at 64: 0.995594, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1743403166.086060 3426407 buffer_comparator.cc:156] Difference at 65: 0.406955, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1743403166.086064 3426407 buffer_comparator.cc:156] Difference at 66: 0.639394, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1743403166.086067 3426407 buffer_comparator.cc:156] Difference at 67: 0.748706, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1743403166.086070 3426407 buffer_comparator.cc:156] Difference at 68: 0.439518, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1743403166.086073 3426407 buffer_comparator.cc:156] Difference at 69: 0.825584, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1743403166.086075 3426407 buffer_comparator.cc:156] Difference at 70: 0.0283883, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1743403166.086078 3426407 buffer_comparator.cc:156] Difference at 71: 0.789963, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1743403166.086081 3426407 buffer_comparator.cc:156] Difference at 72: 0.0831509, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1743403166.086084 3426407 buffer_comparator.cc:156] Difference at 73: 0.318524, expected 17.8359</span></span>
<span class="line"><span>2025-03-31 06:39:26.086089: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.088558 3426407 buffer_comparator.cc:156] Difference at 64: 0.995594, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1743403166.088572 3426407 buffer_comparator.cc:156] Difference at 65: 0.406955, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1743403166.088575 3426407 buffer_comparator.cc:156] Difference at 66: 0.639394, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1743403166.088578 3426407 buffer_comparator.cc:156] Difference at 67: 0.748706, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1743403166.088581 3426407 buffer_comparator.cc:156] Difference at 68: 0.439518, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1743403166.088584 3426407 buffer_comparator.cc:156] Difference at 69: 0.825584, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1743403166.088587 3426407 buffer_comparator.cc:156] Difference at 70: 0.0283883, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1743403166.088590 3426407 buffer_comparator.cc:156] Difference at 71: 0.789963, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1743403166.088593 3426407 buffer_comparator.cc:156] Difference at 72: 0.0831509, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1743403166.088596 3426407 buffer_comparator.cc:156] Difference at 73: 0.318524, expected 17.8359</span></span>
<span class="line"><span>2025-03-31 06:39:26.088600: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403166.091080 3426407 buffer_comparator.cc:156] Difference at 64: 0.995594, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1743403166.091098 3426407 buffer_comparator.cc:156] Difference at 65: 0.406955, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1743403166.091101 3426407 buffer_comparator.cc:156] Difference at 66: 0.639394, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1743403166.091104 3426407 buffer_comparator.cc:156] Difference at 67: 0.748706, expected 17.9799</span></span>
<span class="line"><span>Epoch   1	Train Loss: 15.860073	Train Acc: 20.0000%	Val Loss: 8.312882	Val Acc: 21.0000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 10.085080	Train Acc: 22.8571%	Val Loss: 4.015713	Val Acc: 27.4000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 4.625581	Train Acc: 41.4286%	Val Loss: 1.891236	Val Acc: 40.0000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 2.042694	Train Acc: 50.0000%	Val Loss: 1.987250	Val Acc: 42.8000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.796811	Train Acc: 57.8571%	Val Loss: 1.967420	Val Acc: 44.4000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.552516	Train Acc: 67.1429%	Val Loss: 1.738501	Val Acc: 49.4000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.280803	Train Acc: 68.5714%	Val Loss: 1.581307	Val Acc: 58.0000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.190124	Train Acc: 70.0000%	Val Loss: 1.520821	Val Acc: 59.4000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 1.143173	Train Acc: 72.8571%	Val Loss: 1.487494	Val Acc: 60.6000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 1.018299	Train Acc: 74.2857%	Val Loss: 1.458281	Val Acc: 61.6000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.934400	Train Acc: 77.8571%	Val Loss: 1.433189	Val Acc: 62.4000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.859890	Train Acc: 79.2857%	Val Loss: 1.420378	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.796815	Train Acc: 80.0000%	Val Loss: 1.420816	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.750421	Train Acc: 82.1429%	Val Loss: 1.432797	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.690279	Train Acc: 84.2857%	Val Loss: 1.459450	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.639252	Train Acc: 85.7143%	Val Loss: 1.499520	Val Acc: 63.4000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.607683	Train Acc: 85.7143%	Val Loss: 1.525047	Val Acc: 63.2000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.590651	Train Acc: 86.4286%	Val Loss: 1.522617	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.565949	Train Acc: 87.1429%	Val Loss: 1.501743	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.533952	Train Acc: 87.1429%	Val Loss: 1.481200	Val Acc: 66.2000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.502455	Train Acc: 87.8571%	Val Loss: 1.471943	Val Acc: 67.2000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.479236	Train Acc: 87.8571%	Val Loss: 1.473907	Val Acc: 67.4000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.462111	Train Acc: 87.8571%	Val Loss: 1.484317	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.448312	Train Acc: 87.1429%	Val Loss: 1.503860	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.436555	Train Acc: 87.1429%	Val Loss: 1.530067	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.425669	Train Acc: 88.5714%	Val Loss: 1.561568	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.415378	Train Acc: 89.2857%	Val Loss: 1.593501	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.405305	Train Acc: 89.2857%	Val Loss: 1.623102	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.395369	Train Acc: 89.2857%	Val Loss: 1.646807	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  30	Train Loss: 0.385057	Train Acc: 89.2857%	Val Loss: 1.663199	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  31	Train Loss: 0.374264	Train Acc: 91.4286%	Val Loss: 1.672272	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  32	Train Loss: 0.363127	Train Acc: 92.1429%	Val Loss: 1.674560	Val Acc: 64.2000%</span></span>
<span class="line"><span>Early Stopping at Epoch 32</span></span>
<span class="line"><span>2025-03-31 06:40:30.012473: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:40:30.265502: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:40:30.435223: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:40:30.833966: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743403230.841810 3426407 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743403230.841875 3426407 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743403230.841882 3426407 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743403230.841890 3426407 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.841897 3426407 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743403230.841904 3426407 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743403230.841911 3426407 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.841918 3426407 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743403230.841924 3426407 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.841931 3426407 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-31 06:40:30.841946: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.845567 3426407 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743403230.845600 3426407 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743403230.845608 3426407 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743403230.845615 3426407 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.845621 3426407 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743403230.845628 3426407 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743403230.845635 3426407 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.845642 3426407 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743403230.845648 3426407 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.845655 3426407 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-31 06:40:30.845666: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.849112 3426407 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743403230.849127 3426407 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743403230.849130 3426407 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743403230.849133 3426407 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.849136 3426407 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743403230.849139 3426407 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743403230.849144 3426407 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.849147 3426407 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743403230.849150 3426407 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.849153 3426407 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-31 06:40:30.849158: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.852684 3426407 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743403230.852698 3426407 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743403230.852701 3426407 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743403230.852704 3426407 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.852707 3426407 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743403230.852710 3426407 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743403230.852713 3426407 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.852716 3426407 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743403230.852719 3426407 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.852722 3426407 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-31 06:40:30.852727: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.856272 3426407 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743403230.856285 3426407 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743403230.856288 3426407 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743403230.856291 3426407 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.856294 3426407 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743403230.856298 3426407 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743403230.856301 3426407 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.856304 3426407 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743403230.856306 3426407 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.856309 3426407 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-31 06:40:30.856314: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.859727 3426407 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743403230.859740 3426407 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743403230.859743 3426407 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743403230.859747 3426407 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.859750 3426407 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743403230.859753 3426407 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743403230.859756 3426407 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.859759 3426407 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743403230.859764 3426407 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.859767 3426407 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-31 06:40:30.859771: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.863135 3426407 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1743403230.863148 3426407 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743403230.863152 3426407 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743403230.863155 3426407 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743403230.863158 3426407 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743403230.863161 3426407 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743403230.863164 3426407 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743403230.863167 3426407 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1743403230.863170 3426407 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743403230.863173 3426407 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-03-31 06:40:30.863178: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.866468 3426407 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743403230.866481 3426407 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743403230.866484 3426407 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743403230.866487 3426407 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743403230.866490 3426407 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743403230.866493 3426407 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743403230.866496 3426407 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743403230.866499 3426407 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1743403230.866502 3426407 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1743403230.866505 3426407 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-31 06:40:30.866509: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.869876 3426407 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743403230.869889 3426407 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743403230.869893 3426407 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743403230.869896 3426407 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743403230.869899 3426407 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743403230.869902 3426407 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743403230.869905 3426407 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743403230.869908 3426407 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.869911 3426407 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743403230.869913 3426407 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-31 06:40:30.869920: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.873200 3426407 buffer_comparator.cc:156] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1743403230.873215 3426407 buffer_comparator.cc:156] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1743403230.873219 3426407 buffer_comparator.cc:156] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1743403230.873223 3426407 buffer_comparator.cc:156] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1743403230.873226 3426407 buffer_comparator.cc:156] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1743403230.873230 3426407 buffer_comparator.cc:156] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1743403230.873233 3426407 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743403230.873236 3426407 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743403230.873239 3426407 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743403230.873242 3426407 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-03-31 06:40:30.873247: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.876610 3426407 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743403230.876623 3426407 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743403230.876627 3426407 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743403230.876630 3426407 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743403230.876633 3426407 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743403230.876636 3426407 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743403230.876639 3426407 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743403230.876642 3426407 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.876645 3426407 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743403230.876648 3426407 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-31 06:40:30.876652: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.880059 3426407 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743403230.880072 3426407 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743403230.880075 3426407 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743403230.880079 3426407 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743403230.880081 3426407 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743403230.880084 3426407 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743403230.880088 3426407 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743403230.880090 3426407 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743403230.880093 3426407 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743403230.880096 3426407 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-31 06:40:30.880101: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.883487 3426407 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403230.883500 3426407 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743403230.883503 3426407 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743403230.883507 3426407 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403230.883510 3426407 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743403230.883513 3426407 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.883516 3426407 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.883519 3426407 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743403230.883522 3426407 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.883524 3426407 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-31 06:40:30.883529: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.886906 3426407 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403230.886918 3426407 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743403230.886921 3426407 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743403230.886925 3426407 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403230.886928 3426407 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743403230.886931 3426407 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.886934 3426407 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.886936 3426407 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743403230.886939 3426407 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.886942 3426407 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-31 06:40:30.886947: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.890319 3426407 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403230.890331 3426407 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743403230.890334 3426407 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743403230.890338 3426407 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403230.890341 3426407 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743403230.890344 3426407 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.890347 3426407 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.890350 3426407 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743403230.890353 3426407 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.890356 3426407 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-31 06:40:30.890361: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.893666 3426407 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1743403230.893680 3426407 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403230.893684 3426407 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743403230.893687 3426407 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743403230.893690 3426407 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403230.893693 3426407 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743403230.893696 3426407 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.893699 3426407 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1743403230.893702 3426407 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.893705 3426407 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-03-31 06:40:30.893710: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.896998 3426407 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743403230.897010 3426407 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743403230.897014 3426407 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743403230.897017 3426407 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743403230.897020 3426407 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743403230.897023 3426407 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743403230.897026 3426407 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.897029 3426407 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743403230.897032 3426407 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743403230.897035 3426407 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-31 06:40:30.897039: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.900594 3426407 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743403230.900607 3426407 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743403230.900610 3426407 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743403230.900613 3426407 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.900616 3426407 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743403230.900619 3426407 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743403230.900622 3426407 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743403230.900625 3426407 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1743403230.900628 3426407 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1743403230.900631 3426407 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-03-31 06:40:30.900636: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403230.904180 3426407 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743403230.904196 3426407 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743403230.904200 3426407 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743403230.904213 3426407 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743403230.904216 3426407 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743403230.904219 3426407 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743403230.904222 3426407 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743403230.904225 3426407 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1743403230.904228 3426407 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1743403230.904231 3426407 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-03-31 06:40:30.904237: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-03-31 06:40:32.148882: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:40:32.401010: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:40:32.467732: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-31 06:40:33.088901: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743403233.096273 3426407 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743403233.096325 3426407 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743403233.096333 3426407 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743403233.096341 3426407 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.096348 3426407 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743403233.096354 3426407 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743403233.096361 3426407 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743403233.096367 3426407 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743403233.096374 3426407 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.096380 3426407 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-31 06:40:33.096394: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.100164 3426407 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743403233.100190 3426407 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743403233.100198 3426407 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743403233.100205 3426407 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.100211 3426407 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743403233.100218 3426407 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743403233.100224 3426407 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743403233.100231 3426407 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743403233.100237 3426407 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.100246 3426407 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-31 06:40:33.100257: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.103855 3426407 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743403233.103871 3426407 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743403233.103874 3426407 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743403233.103877 3426407 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.103880 3426407 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743403233.103883 3426407 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743403233.103886 3426407 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743403233.103889 3426407 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743403233.103892 3426407 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.103894 3426407 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-31 06:40:33.103899: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.107392 3426407 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743403233.107406 3426407 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743403233.107410 3426407 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743403233.107412 3426407 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.107415 3426407 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743403233.107418 3426407 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743403233.107421 3426407 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743403233.107424 3426407 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743403233.107427 3426407 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.107430 3426407 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-31 06:40:33.107434: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.110795 3426407 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743403233.110811 3426407 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743403233.110814 3426407 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743403233.110817 3426407 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.110820 3426407 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743403233.110823 3426407 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743403233.110825 3426407 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743403233.110828 3426407 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743403233.110831 3426407 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.110834 3426407 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-31 06:40:33.110839: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.114351 3426407 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743403233.114368 3426407 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743403233.114372 3426407 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743403233.114375 3426407 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.114378 3426407 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743403233.114381 3426407 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743403233.114385 3426407 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743403233.114389 3426407 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743403233.114392 3426407 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.114395 3426407 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-31 06:40:33.114400: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.117741 3426407 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743403233.117753 3426407 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743403233.117756 3426407 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743403233.117759 3426407 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743403233.117762 3426407 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743403233.117765 3426407 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743403233.117768 3426407 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743403233.117771 3426407 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743403233.117774 3426407 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743403233.117777 3426407 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-31 06:40:33.117781: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.121072 3426407 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743403233.121086 3426407 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743403233.121089 3426407 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743403233.121092 3426407 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743403233.121095 3426407 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743403233.121098 3426407 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743403233.121101 3426407 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743403233.121103 3426407 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743403233.121106 3426407 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743403233.121109 3426407 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-31 06:40:33.121114: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.124472 3426407 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743403233.124486 3426407 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743403233.124489 3426407 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743403233.124492 3426407 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743403233.124495 3426407 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743403233.124500 3426407 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743403233.124503 3426407 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743403233.124506 3426407 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743403233.124509 3426407 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743403233.124512 3426407 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-31 06:40:33.124516: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.127786 3426407 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743403233.127799 3426407 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743403233.127803 3426407 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743403233.127806 3426407 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743403233.127809 3426407 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743403233.127811 3426407 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743403233.127814 3426407 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743403233.127817 3426407 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743403233.127820 3426407 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743403233.127823 3426407 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-31 06:40:33.127828: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.131203 3426407 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743403233.131217 3426407 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743403233.131221 3426407 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743403233.131224 3426407 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743403233.131226 3426407 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743403233.131229 3426407 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743403233.131232 3426407 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743403233.131235 3426407 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743403233.131238 3426407 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743403233.131241 3426407 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-31 06:40:33.131245: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.134641 3426407 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743403233.134655 3426407 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743403233.134658 3426407 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743403233.134661 3426407 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743403233.134664 3426407 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743403233.134667 3426407 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743403233.134670 3426407 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743403233.134673 3426407 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743403233.134676 3426407 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743403233.134680 3426407 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-31 06:40:33.134685: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.138076 3426407 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743403233.138089 3426407 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.138093 3426407 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.138096 3426407 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743403233.138098 3426407 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743403233.138101 3426407 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743403233.138104 3426407 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743403233.138107 3426407 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743403233.138110 3426407 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743403233.138113 3426407 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-31 06:40:33.138117: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.141479 3426407 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743403233.141491 3426407 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.141494 3426407 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.141497 3426407 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743403233.141500 3426407 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743403233.141503 3426407 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743403233.141506 3426407 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743403233.141508 3426407 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743403233.141511 3426407 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743403233.141514 3426407 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-31 06:40:33.141519: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.144897 3426407 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743403233.144911 3426407 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.144914 3426407 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.144917 3426407 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743403233.144920 3426407 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743403233.144923 3426407 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743403233.144926 3426407 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743403233.144929 3426407 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743403233.144931 3426407 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743403233.144934 3426407 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-31 06:40:33.144939: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.148243 3426407 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743403233.148257 3426407 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.148261 3426407 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.148264 3426407 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743403233.148267 3426407 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743403233.148269 3426407 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743403233.148272 3426407 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743403233.148275 3426407 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743403233.148278 3426407 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743403233.148281 3426407 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-31 06:40:33.148285: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.151588 3426407 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743403233.151601 3426407 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743403233.151604 3426407 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743403233.151607 3426407 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743403233.151610 3426407 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743403233.151613 3426407 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743403233.151616 3426407 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743403233.151618 3426407 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743403233.151621 3426407 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743403233.151624 3426407 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-31 06:40:33.151629: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.155171 3426407 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743403233.155184 3426407 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743403233.155188 3426407 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743403233.155191 3426407 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743403233.155193 3426407 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743403233.155196 3426407 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743403233.155199 3426407 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743403233.155202 3426407 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743403233.155205 3426407 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743403233.155208 3426407 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-31 06:40:33.155213: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743403233.158771 3426407 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743403233.158784 3426407 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743403233.158787 3426407 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743403233.158790 3426407 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743403233.158793 3426407 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743403233.158798 3426407 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743403233.158801 3426407 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743403233.158803 3426407 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743403233.158806 3426407 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743403233.158809 3426407 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-31 06:40:33.158814: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.515153	Test Acc: 65.8000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
