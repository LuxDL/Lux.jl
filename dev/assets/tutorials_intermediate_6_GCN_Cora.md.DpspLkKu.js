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
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-04-07 02:14:05.560411: I external/xla/xla/service/service.cc:152] XLA service 0x68ae0b0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-04-07 02:14:05.560766: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743992045.562879 3890942 se_gpu_pjrt_client.cc:1040] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743992045.563094 3890942 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743992045.563251 3890942 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743992045.579360 3890942 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-8/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-8/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-04-07 02:15:09.497044: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:10.232957: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:10.559650: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:10.786513: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 616 bytes spill stores, 616 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:11.110808: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 32 bytes spill stores, 32 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:11.169944: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:11.447441: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:11.451777: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 268 bytes spill stores, 268 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:11.628029: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:12.362223: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:12.440467: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:12.488698: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:12.511266: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 116 bytes spill stores, 116 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:13.756779: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:13.795470: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 52 bytes spill stores, 52 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:13.825868: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:14.383395: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:15.247618: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:15:15.423463: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 256 bytes spill stores, 256 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743992115.572320 3890942 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.573124 3890942 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743992115.573132 3890942 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743992115.573139 3890942 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743992115.573146 3890942 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743992115.573153 3890942 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743992115.573160 3890942 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992115.573167 3890942 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743992115.573174 3890942 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743992115.573180 3890942 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-07 02:15:15.573197: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.575936 3890942 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.575955 3890942 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743992115.575960 3890942 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743992115.575964 3890942 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743992115.575968 3890942 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743992115.575972 3890942 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743992115.575977 3890942 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992115.575981 3890942 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743992115.575985 3890942 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743992115.575989 3890942 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-07 02:15:15.575996: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.578223 3890942 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.578245 3890942 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743992115.578249 3890942 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743992115.578254 3890942 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743992115.578258 3890942 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743992115.578262 3890942 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743992115.578266 3890942 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992115.578271 3890942 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743992115.578275 3890942 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743992115.578279 3890942 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-07 02:15:15.578299: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.580589 3890942 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.580609 3890942 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743992115.580613 3890942 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743992115.580617 3890942 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743992115.580622 3890942 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743992115.580626 3890942 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743992115.580630 3890942 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992115.580634 3890942 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743992115.580639 3890942 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743992115.580643 3890942 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-07 02:15:15.580649: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.582907 3890942 buffer_comparator.cc:156] Difference at 0: 1140.51, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743992115.582926 3890942 buffer_comparator.cc:156] Difference at 1: 1405.76, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743992115.582930 3890942 buffer_comparator.cc:156] Difference at 2: 2133.59, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743992115.582934 3890942 buffer_comparator.cc:156] Difference at 3: 1840.1, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743992115.582939 3890942 buffer_comparator.cc:156] Difference at 4: 1308.3, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743992115.582943 3890942 buffer_comparator.cc:156] Difference at 5: 2065.73, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.582947 3890942 buffer_comparator.cc:156] Difference at 6: 1481.85, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1743992115.582951 3890942 buffer_comparator.cc:156] Difference at 7: 1113.93, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743992115.582956 3890942 buffer_comparator.cc:156] Difference at 8: 1359.57, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743992115.582960 3890942 buffer_comparator.cc:156] Difference at 9: 2049.53, expected 1833.76</span></span>
<span class="line"><span>2025-04-07 02:15:15.582966: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.585236 3890942 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.585254 3890942 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743992115.585258 3890942 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743992115.585263 3890942 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743992115.585267 3890942 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743992115.585271 3890942 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743992115.585275 3890942 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992115.585280 3890942 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743992115.585284 3890942 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743992115.585288 3890942 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-04-07 02:15:15.585295: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.587545 3890942 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743992115.587561 3890942 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743992115.587564 3890942 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743992115.587567 3890942 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743992115.587570 3890942 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743992115.587573 3890942 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743992115.587576 3890942 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743992115.587579 3890942 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743992115.587583 3890942 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743992115.587586 3890942 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-04-07 02:15:15.587590: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.589727 3890942 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743992115.589740 3890942 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743992115.589743 3890942 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743992115.589746 3890942 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743992115.589749 3890942 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743992115.589752 3890942 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743992115.589756 3890942 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743992115.589758 3890942 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743992115.589762 3890942 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743992115.589764 3890942 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-04-07 02:15:15.589769: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.591909 3890942 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743992115.591924 3890942 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743992115.591927 3890942 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743992115.591930 3890942 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743992115.591933 3890942 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.591936 3890942 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.591939 3890942 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743992115.591942 3890942 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743992115.591945 3890942 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743992115.591948 3890942 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-04-07 02:15:15.591953: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.594088 3890942 buffer_comparator.cc:156] Difference at 0: 1057.99, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743992115.594102 3890942 buffer_comparator.cc:156] Difference at 1: 1320.07, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743992115.594106 3890942 buffer_comparator.cc:156] Difference at 2: 2005.81, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743992115.594109 3890942 buffer_comparator.cc:156] Difference at 3: 1746.91, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743992115.594112 3890942 buffer_comparator.cc:156] Difference at 4: 1253.09, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743992115.594115 3890942 buffer_comparator.cc:156] Difference at 7: 1176.41, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743992115.594118 3890942 buffer_comparator.cc:156] Difference at 8: 1399.66, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743992115.594121 3890942 buffer_comparator.cc:156] Difference at 9: 2127.08, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1743992115.594124 3890942 buffer_comparator.cc:156] Difference at 10: 1879.74, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1743992115.594127 3890942 buffer_comparator.cc:156] Difference at 11: 1363.62, expected 1121.95</span></span>
<span class="line"><span>2025-04-07 02:15:15.594131: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.596277 3890942 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743992115.596292 3890942 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743992115.596295 3890942 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743992115.596298 3890942 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743992115.596301 3890942 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.596304 3890942 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.596307 3890942 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743992115.596310 3890942 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743992115.596313 3890942 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743992115.596316 3890942 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-04-07 02:15:15.596321: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.598462 3890942 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743992115.598475 3890942 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743992115.598478 3890942 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743992115.598481 3890942 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743992115.598484 3890942 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.598487 3890942 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.598490 3890942 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743992115.598493 3890942 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743992115.598496 3890942 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743992115.598499 3890942 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-04-07 02:15:15.598504: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.600697 3890942 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743992115.600710 3890942 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992115.600713 3890942 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.600718 3890942 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743992115.600721 3890942 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992115.600724 3890942 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.600727 3890942 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.600730 3890942 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743992115.600733 3890942 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743992115.600736 3890942 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-07 02:15:15.600740: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.602906 3890942 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743992115.602926 3890942 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992115.602929 3890942 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.602932 3890942 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743992115.602935 3890942 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992115.602938 3890942 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.602941 3890942 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.602944 3890942 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743992115.602947 3890942 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743992115.602950 3890942 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-07 02:15:15.602955: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.605159 3890942 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743992115.605172 3890942 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992115.605176 3890942 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.605179 3890942 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743992115.605182 3890942 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992115.605185 3890942 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.605188 3890942 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.605191 3890942 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743992115.605194 3890942 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743992115.605197 3890942 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-07 02:15:15.605201: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.607353 3890942 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743992115.607368 3890942 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992115.607371 3890942 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.607374 3890942 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743992115.607377 3890942 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992115.607383 3890942 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.607386 3890942 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.607389 3890942 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743992115.607392 3890942 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743992115.607395 3890942 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-07 02:15:15.607399: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.609525 3890942 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743992115.609538 3890942 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992115.609541 3890942 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743992115.609544 3890942 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743992115.609547 3890942 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992115.609550 3890942 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.609553 3890942 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992115.609556 3890942 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743992115.609559 3890942 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743992115.609562 3890942 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-04-07 02:15:15.609567: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.611869 3890942 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743992115.611884 3890942 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743992115.611887 3890942 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.611890 3890942 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743992115.611893 3890942 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743992115.611896 3890942 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743992115.611899 3890942 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743992115.611902 3890942 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743992115.611905 3890942 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743992115.611908 3890942 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-04-07 02:15:15.611913: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.614212 3890942 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743992115.614225 3890942 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743992115.614229 3890942 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743992115.614232 3890942 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743992115.614235 3890942 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743992115.614238 3890942 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743992115.614241 3890942 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743992115.614246 3890942 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743992115.614249 3890942 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743992115.614252 3890942 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-04-07 02:15:15.614257: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.616821 3890942 buffer_comparator.cc:156] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743992115.616834 3890942 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743992115.616837 3890942 buffer_comparator.cc:156] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743992115.616840 3890942 buffer_comparator.cc:156] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743992115.616843 3890942 buffer_comparator.cc:156] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743992115.616845 3890942 buffer_comparator.cc:156] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743992115.616848 3890942 buffer_comparator.cc:156] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743992115.616851 3890942 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743992115.616853 3890942 buffer_comparator.cc:156] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743992115.616856 3890942 buffer_comparator.cc:156] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-04-07 02:15:15.616861: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.618389 3890942 buffer_comparator.cc:156] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743992115.618409 3890942 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743992115.618412 3890942 buffer_comparator.cc:156] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743992115.618415 3890942 buffer_comparator.cc:156] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743992115.618418 3890942 buffer_comparator.cc:156] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743992115.618421 3890942 buffer_comparator.cc:156] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743992115.618423 3890942 buffer_comparator.cc:156] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743992115.618426 3890942 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743992115.618429 3890942 buffer_comparator.cc:156] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743992115.618432 3890942 buffer_comparator.cc:156] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-04-07 02:15:15.618436: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.619958 3890942 buffer_comparator.cc:156] Difference at 16: -nan, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743992115.619971 3890942 buffer_comparator.cc:156] Difference at 17: -nan, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743992115.619974 3890942 buffer_comparator.cc:156] Difference at 18: -nan, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743992115.619977 3890942 buffer_comparator.cc:156] Difference at 19: -nan, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743992115.619979 3890942 buffer_comparator.cc:156] Difference at 20: -nan, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743992115.619982 3890942 buffer_comparator.cc:156] Difference at 21: -nan, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743992115.619985 3890942 buffer_comparator.cc:156] Difference at 22: -nan, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743992115.619987 3890942 buffer_comparator.cc:156] Difference at 23: -nan, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743992115.619990 3890942 buffer_comparator.cc:156] Difference at 24: -nan, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743992115.619993 3890942 buffer_comparator.cc:156] Difference at 25: -nan, expected 13.4166</span></span>
<span class="line"><span>2025-04-07 02:15:15.619999: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.621532 3890942 buffer_comparator.cc:156] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1743992115.621545 3890942 buffer_comparator.cc:156] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1743992115.621548 3890942 buffer_comparator.cc:156] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1743992115.621550 3890942 buffer_comparator.cc:156] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1743992115.621553 3890942 buffer_comparator.cc:156] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1743992115.621556 3890942 buffer_comparator.cc:156] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1743992115.621558 3890942 buffer_comparator.cc:156] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1743992115.621561 3890942 buffer_comparator.cc:156] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1743992115.621564 3890942 buffer_comparator.cc:156] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1743992115.621567 3890942 buffer_comparator.cc:156] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-04-07 02:15:15.621571: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.623112 3890942 buffer_comparator.cc:156] Difference at 32: -nan, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1743992115.623125 3890942 buffer_comparator.cc:156] Difference at 33: -nan, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1743992115.623128 3890942 buffer_comparator.cc:156] Difference at 34: -nan, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1743992115.623131 3890942 buffer_comparator.cc:156] Difference at 35: -nan, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1743992115.623134 3890942 buffer_comparator.cc:156] Difference at 36: -nan, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1743992115.623136 3890942 buffer_comparator.cc:156] Difference at 37: -nan, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1743992115.623139 3890942 buffer_comparator.cc:156] Difference at 38: -nan, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1743992115.623142 3890942 buffer_comparator.cc:156] Difference at 39: -nan, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1743992115.623145 3890942 buffer_comparator.cc:156] Difference at 40: -nan, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1743992115.623147 3890942 buffer_comparator.cc:156] Difference at 41: -nan, expected 13.7427</span></span>
<span class="line"><span>2025-04-07 02:15:15.623152: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.624679 3890942 buffer_comparator.cc:156] Difference at 64: -nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743992115.624691 3890942 buffer_comparator.cc:156] Difference at 65: -nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743992115.624694 3890942 buffer_comparator.cc:156] Difference at 66: -nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743992115.624697 3890942 buffer_comparator.cc:156] Difference at 67: -nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743992115.624700 3890942 buffer_comparator.cc:156] Difference at 68: -nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743992115.624702 3890942 buffer_comparator.cc:156] Difference at 69: -nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743992115.624705 3890942 buffer_comparator.cc:156] Difference at 70: -nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743992115.624708 3890942 buffer_comparator.cc:156] Difference at 71: -nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743992115.624711 3890942 buffer_comparator.cc:156] Difference at 72: -nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743992115.624713 3890942 buffer_comparator.cc:156] Difference at 73: -nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-07 02:15:15.624718: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.626278 3890942 buffer_comparator.cc:156] Difference at 0: 16.5369, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1743992115.626295 3890942 buffer_comparator.cc:156] Difference at 1: 19.4176, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1743992115.626298 3890942 buffer_comparator.cc:156] Difference at 2: 16.204, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1743992115.626301 3890942 buffer_comparator.cc:156] Difference at 6: 13.1759, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1743992115.626304 3890942 buffer_comparator.cc:156] Difference at 9: 16.3002, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1743992115.626307 3890942 buffer_comparator.cc:156] Difference at 11: 15.6508, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1743992115.626310 3890942 buffer_comparator.cc:156] Difference at 12: 20.6885, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1743992115.626313 3890942 buffer_comparator.cc:156] Difference at 13: 17.247, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1743992115.626316 3890942 buffer_comparator.cc:156] Difference at 14: 14.7694, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1743992115.626319 3890942 buffer_comparator.cc:156] Difference at 16: 17.2743, expected 15.1227</span></span>
<span class="line"><span>2025-04-07 02:15:15.626324: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.627999 3890942 buffer_comparator.cc:156] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743992115.628018 3890942 buffer_comparator.cc:156] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743992115.628021 3890942 buffer_comparator.cc:156] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743992115.628024 3890942 buffer_comparator.cc:156] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743992115.628026 3890942 buffer_comparator.cc:156] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743992115.628029 3890942 buffer_comparator.cc:156] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743992115.628032 3890942 buffer_comparator.cc:156] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743992115.628035 3890942 buffer_comparator.cc:156] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743992115.628037 3890942 buffer_comparator.cc:156] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743992115.628040 3890942 buffer_comparator.cc:156] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-07 02:15:15.628044: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.629580 3890942 buffer_comparator.cc:156] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743992115.629594 3890942 buffer_comparator.cc:156] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743992115.629597 3890942 buffer_comparator.cc:156] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743992115.629599 3890942 buffer_comparator.cc:156] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743992115.629602 3890942 buffer_comparator.cc:156] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743992115.629605 3890942 buffer_comparator.cc:156] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743992115.629608 3890942 buffer_comparator.cc:156] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743992115.629610 3890942 buffer_comparator.cc:156] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743992115.629613 3890942 buffer_comparator.cc:156] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743992115.629616 3890942 buffer_comparator.cc:156] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-07 02:15:15.629621: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.631184 3890942 buffer_comparator.cc:156] Difference at 64: nan, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743992115.631201 3890942 buffer_comparator.cc:156] Difference at 65: nan, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743992115.631204 3890942 buffer_comparator.cc:156] Difference at 66: nan, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743992115.631206 3890942 buffer_comparator.cc:156] Difference at 67: nan, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743992115.631211 3890942 buffer_comparator.cc:156] Difference at 68: nan, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743992115.631214 3890942 buffer_comparator.cc:156] Difference at 69: nan, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743992115.631217 3890942 buffer_comparator.cc:156] Difference at 70: nan, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743992115.631219 3890942 buffer_comparator.cc:156] Difference at 71: nan, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743992115.631222 3890942 buffer_comparator.cc:156] Difference at 72: nan, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743992115.631225 3890942 buffer_comparator.cc:156] Difference at 73: nan, expected 14.1923</span></span>
<span class="line"><span>2025-04-07 02:15:15.631229: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.632757 3890942 buffer_comparator.cc:156] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743992115.632771 3890942 buffer_comparator.cc:156] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743992115.632774 3890942 buffer_comparator.cc:156] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743992115.632777 3890942 buffer_comparator.cc:156] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743992115.632779 3890942 buffer_comparator.cc:156] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743992115.632782 3890942 buffer_comparator.cc:156] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743992115.632785 3890942 buffer_comparator.cc:156] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743992115.632787 3890942 buffer_comparator.cc:156] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743992115.632790 3890942 buffer_comparator.cc:156] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743992115.632793 3890942 buffer_comparator.cc:156] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-04-07 02:15:15.632797: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.634331 3890942 buffer_comparator.cc:156] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743992115.634344 3890942 buffer_comparator.cc:156] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743992115.634347 3890942 buffer_comparator.cc:156] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743992115.634350 3890942 buffer_comparator.cc:156] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743992115.634353 3890942 buffer_comparator.cc:156] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743992115.634355 3890942 buffer_comparator.cc:156] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743992115.634358 3890942 buffer_comparator.cc:156] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743992115.634361 3890942 buffer_comparator.cc:156] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743992115.634363 3890942 buffer_comparator.cc:156] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743992115.634366 3890942 buffer_comparator.cc:156] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-04-07 02:15:15.634371: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.635916 3890942 buffer_comparator.cc:156] Difference at 128: nan, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743992115.635930 3890942 buffer_comparator.cc:156] Difference at 129: nan, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743992115.635933 3890942 buffer_comparator.cc:156] Difference at 130: nan, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743992115.635935 3890942 buffer_comparator.cc:156] Difference at 131: nan, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743992115.635938 3890942 buffer_comparator.cc:156] Difference at 132: nan, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743992115.635941 3890942 buffer_comparator.cc:156] Difference at 133: nan, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743992115.635943 3890942 buffer_comparator.cc:156] Difference at 134: nan, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743992115.635948 3890942 buffer_comparator.cc:156] Difference at 135: nan, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743992115.635951 3890942 buffer_comparator.cc:156] Difference at 136: nan, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743992115.635953 3890942 buffer_comparator.cc:156] Difference at 137: nan, expected 12.9584</span></span>
<span class="line"><span>2025-04-07 02:15:15.635958: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.637505 3890942 buffer_comparator.cc:156] Difference at 256: nan, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743992115.637523 3890942 buffer_comparator.cc:156] Difference at 257: nan, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743992115.637527 3890942 buffer_comparator.cc:156] Difference at 258: nan, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743992115.637529 3890942 buffer_comparator.cc:156] Difference at 259: nan, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743992115.637532 3890942 buffer_comparator.cc:156] Difference at 260: nan, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743992115.637535 3890942 buffer_comparator.cc:156] Difference at 261: nan, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743992115.637538 3890942 buffer_comparator.cc:156] Difference at 262: nan, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743992115.637540 3890942 buffer_comparator.cc:156] Difference at 263: nan, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743992115.637543 3890942 buffer_comparator.cc:156] Difference at 264: nan, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743992115.637546 3890942 buffer_comparator.cc:156] Difference at 265: nan, expected 15.766</span></span>
<span class="line"><span>2025-04-07 02:15:15.637550: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.639112 3890942 buffer_comparator.cc:156] Difference at 256: nan, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743992115.639125 3890942 buffer_comparator.cc:156] Difference at 257: nan, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743992115.639128 3890942 buffer_comparator.cc:156] Difference at 258: nan, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743992115.639131 3890942 buffer_comparator.cc:156] Difference at 259: nan, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743992115.639133 3890942 buffer_comparator.cc:156] Difference at 260: nan, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743992115.639136 3890942 buffer_comparator.cc:156] Difference at 261: nan, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743992115.639139 3890942 buffer_comparator.cc:156] Difference at 262: nan, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743992115.639141 3890942 buffer_comparator.cc:156] Difference at 263: nan, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743992115.639144 3890942 buffer_comparator.cc:156] Difference at 264: nan, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743992115.639147 3890942 buffer_comparator.cc:156] Difference at 265: nan, expected 15.766</span></span>
<span class="line"><span>2025-04-07 02:15:15.639151: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.640694 3890942 buffer_comparator.cc:156] Difference at 256: nan, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743992115.640709 3890942 buffer_comparator.cc:156] Difference at 257: nan, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743992115.640712 3890942 buffer_comparator.cc:156] Difference at 258: nan, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743992115.640715 3890942 buffer_comparator.cc:156] Difference at 259: nan, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743992115.640718 3890942 buffer_comparator.cc:156] Difference at 260: nan, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743992115.640720 3890942 buffer_comparator.cc:156] Difference at 261: nan, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743992115.640723 3890942 buffer_comparator.cc:156] Difference at 262: nan, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743992115.640726 3890942 buffer_comparator.cc:156] Difference at 263: nan, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743992115.640728 3890942 buffer_comparator.cc:156] Difference at 264: nan, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743992115.640731 3890942 buffer_comparator.cc:156] Difference at 265: nan, expected 15.766</span></span>
<span class="line"><span>2025-04-07 02:15:15.640737: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.642295 3890942 buffer_comparator.cc:156] Difference at 256: nan, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743992115.642308 3890942 buffer_comparator.cc:156] Difference at 257: nan, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743992115.642311 3890942 buffer_comparator.cc:156] Difference at 258: nan, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743992115.642314 3890942 buffer_comparator.cc:156] Difference at 259: nan, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743992115.642317 3890942 buffer_comparator.cc:156] Difference at 260: nan, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743992115.642319 3890942 buffer_comparator.cc:156] Difference at 261: nan, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743992115.642322 3890942 buffer_comparator.cc:156] Difference at 262: nan, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743992115.642325 3890942 buffer_comparator.cc:156] Difference at 263: nan, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743992115.642327 3890942 buffer_comparator.cc:156] Difference at 264: nan, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743992115.642330 3890942 buffer_comparator.cc:156] Difference at 265: nan, expected 15.766</span></span>
<span class="line"><span>2025-04-07 02:15:15.642335: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.646486 3890942 buffer_comparator.cc:156] Difference at 16: -nan, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1743992115.646501 3890942 buffer_comparator.cc:156] Difference at 17: -nan, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1743992115.646504 3890942 buffer_comparator.cc:156] Difference at 18: -nan, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1743992115.646506 3890942 buffer_comparator.cc:156] Difference at 19: -nan, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1743992115.646509 3890942 buffer_comparator.cc:156] Difference at 20: -nan, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1743992115.646512 3890942 buffer_comparator.cc:156] Difference at 21: -nan, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1743992115.646515 3890942 buffer_comparator.cc:156] Difference at 22: -nan, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1743992115.646517 3890942 buffer_comparator.cc:156] Difference at 23: -nan, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1743992115.646520 3890942 buffer_comparator.cc:156] Difference at 24: -nan, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1743992115.646523 3890942 buffer_comparator.cc:156] Difference at 25: -nan, expected 357.082</span></span>
<span class="line"><span>2025-04-07 02:15:15.646527: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.649665 3890942 buffer_comparator.cc:156] Difference at 16: -nan, expected 363.619</span></span>
<span class="line"><span>E0000 00:00:1743992115.649679 3890942 buffer_comparator.cc:156] Difference at 17: -nan, expected 368.882</span></span>
<span class="line"><span>E0000 00:00:1743992115.649682 3890942 buffer_comparator.cc:156] Difference at 18: -nan, expected 358.37</span></span>
<span class="line"><span>E0000 00:00:1743992115.649684 3890942 buffer_comparator.cc:156] Difference at 19: -nan, expected 346.727</span></span>
<span class="line"><span>E0000 00:00:1743992115.649687 3890942 buffer_comparator.cc:156] Difference at 20: -nan, expected 356.216</span></span>
<span class="line"><span>E0000 00:00:1743992115.649690 3890942 buffer_comparator.cc:156] Difference at 21: -nan, expected 358.962</span></span>
<span class="line"><span>E0000 00:00:1743992115.649693 3890942 buffer_comparator.cc:156] Difference at 22: -nan, expected 359.155</span></span>
<span class="line"><span>E0000 00:00:1743992115.649696 3890942 buffer_comparator.cc:156] Difference at 23: -nan, expected 360.559</span></span>
<span class="line"><span>E0000 00:00:1743992115.649698 3890942 buffer_comparator.cc:156] Difference at 24: -nan, expected 371.461</span></span>
<span class="line"><span>E0000 00:00:1743992115.649701 3890942 buffer_comparator.cc:156] Difference at 25: -nan, expected 357.082</span></span>
<span class="line"><span>2025-04-07 02:15:15.649705: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.652214 3890942 buffer_comparator.cc:156] Difference at 64: -nan, expected 357.295</span></span>
<span class="line"><span>E0000 00:00:1743992115.652230 3890942 buffer_comparator.cc:156] Difference at 65: -nan, expected 365.079</span></span>
<span class="line"><span>E0000 00:00:1743992115.652233 3890942 buffer_comparator.cc:156] Difference at 66: -nan, expected 364.297</span></span>
<span class="line"><span>E0000 00:00:1743992115.652236 3890942 buffer_comparator.cc:156] Difference at 67: -nan, expected 356.584</span></span>
<span class="line"><span>E0000 00:00:1743992115.652239 3890942 buffer_comparator.cc:156] Difference at 68: -nan, expected 350.44</span></span>
<span class="line"><span>E0000 00:00:1743992115.652242 3890942 buffer_comparator.cc:156] Difference at 69: -nan, expected 355.742</span></span>
<span class="line"><span>E0000 00:00:1743992115.652244 3890942 buffer_comparator.cc:156] Difference at 70: -nan, expected 347.459</span></span>
<span class="line"><span>E0000 00:00:1743992115.652247 3890942 buffer_comparator.cc:156] Difference at 71: -nan, expected 364.613</span></span>
<span class="line"><span>E0000 00:00:1743992115.652250 3890942 buffer_comparator.cc:156] Difference at 72: -nan, expected 362.734</span></span>
<span class="line"><span>E0000 00:00:1743992115.652252 3890942 buffer_comparator.cc:156] Difference at 73: -nan, expected 362.087</span></span>
<span class="line"><span>2025-04-07 02:15:15.652257: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.654560 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.654580 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.654583 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.654585 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.654588 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.654591 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.654594 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.654596 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.654599 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.654602 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.654606: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.656923 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.656938 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.656941 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.656944 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.656947 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.656950 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.656952 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.656955 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.656958 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.656960 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.656965: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.659206 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.659219 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.659222 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.659225 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.659230 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.659233 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.659236 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.659238 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.659241 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.659244 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.659248: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.661568 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.661582 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.661585 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.661588 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.661591 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.661594 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.661596 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.661599 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.661602 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.661604 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.661609: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.663812 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.663827 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.663830 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.663833 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.663836 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.663839 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.663841 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.663844 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.663847 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.663850 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.663854: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.666183 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.666197 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.666200 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.666203 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.666205 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.666208 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.666211 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.666215 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.666218 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.666221 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.666225: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.668554 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.668567 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.668570 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.668573 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.668576 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.668578 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.668581 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.668584 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.668587 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.668589 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.668594: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.670838 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.670855 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.670858 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.670861 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.670864 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.670867 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.670869 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.670872 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.670875 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>E0000 00:00:1743992115.670877 3890942 buffer_comparator.cc:156] Difference at 137: -nan, expected 357.638</span></span>
<span class="line"><span>2025-04-07 02:15:15.670882: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992115.673223 3890942 buffer_comparator.cc:156] Difference at 128: -nan, expected 364.049</span></span>
<span class="line"><span>E0000 00:00:1743992115.673236 3890942 buffer_comparator.cc:156] Difference at 129: -nan, expected 361.655</span></span>
<span class="line"><span>E0000 00:00:1743992115.673239 3890942 buffer_comparator.cc:156] Difference at 130: -nan, expected 358.35</span></span>
<span class="line"><span>E0000 00:00:1743992115.673241 3890942 buffer_comparator.cc:156] Difference at 131: -nan, expected 355.757</span></span>
<span class="line"><span>E0000 00:00:1743992115.673244 3890942 buffer_comparator.cc:156] Difference at 132: -nan, expected 352.498</span></span>
<span class="line"><span>E0000 00:00:1743992115.673247 3890942 buffer_comparator.cc:156] Difference at 133: -nan, expected 344.673</span></span>
<span class="line"><span>E0000 00:00:1743992115.673250 3890942 buffer_comparator.cc:156] Difference at 134: -nan, expected 365.306</span></span>
<span class="line"><span>E0000 00:00:1743992115.673252 3890942 buffer_comparator.cc:156] Difference at 135: -nan, expected 365.162</span></span>
<span class="line"><span>E0000 00:00:1743992115.673255 3890942 buffer_comparator.cc:156] Difference at 136: -nan, expected 366.674</span></span>
<span class="line"><span>Epoch   1	Train Loss: 15.682838	Train Acc: 23.5714%	Val Loss: 7.225888	Val Acc: 24.2000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 8.972753	Train Acc: 23.5714%	Val Loss: 2.933258	Val Acc: 30.0000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.296663	Train Acc: 45.7143%	Val Loss: 2.069067	Val Acc: 41.0000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 2.021095	Train Acc: 50.7143%	Val Loss: 2.115174	Val Acc: 38.4000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.788062	Train Acc: 61.4286%	Val Loss: 1.902024	Val Acc: 45.6000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.566618	Train Acc: 67.8571%	Val Loss: 1.679744	Val Acc: 54.2000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.411085	Train Acc: 72.8571%	Val Loss: 1.573923	Val Acc: 59.2000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.269617	Train Acc: 75.7143%	Val Loss: 1.567219	Val Acc: 59.8000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 1.148809	Train Acc: 76.4286%	Val Loss: 1.599442	Val Acc: 57.8000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 1.086010	Train Acc: 75.7143%	Val Loss: 1.616633	Val Acc: 58.0000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 1.034065	Train Acc: 77.8571%	Val Loss: 1.594874	Val Acc: 58.2000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.943737	Train Acc: 78.5714%	Val Loss: 1.551072	Val Acc: 61.2000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.831981	Train Acc: 82.8571%	Val Loss: 1.522378	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.739848	Train Acc: 85.0000%	Val Loss: 1.536446	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.700118	Train Acc: 84.2857%	Val Loss: 1.579480	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.680700	Train Acc: 84.2857%	Val Loss: 1.622118	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.650622	Train Acc: 85.0000%	Val Loss: 1.646721	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.607004	Train Acc: 85.0000%	Val Loss: 1.652151	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.570613	Train Acc: 85.0000%	Val Loss: 1.651109	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.543632	Train Acc: 85.7143%	Val Loss: 1.649418	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.519483	Train Acc: 86.4286%	Val Loss: 1.651658	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.496536	Train Acc: 87.8571%	Val Loss: 1.660364	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.474681	Train Acc: 87.1429%	Val Loss: 1.675251	Val Acc: 65.2000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.454312	Train Acc: 87.1429%	Val Loss: 1.694376	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.435466	Train Acc: 87.1429%	Val Loss: 1.714841	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.417872	Train Acc: 88.5714%	Val Loss: 1.734292	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.401149	Train Acc: 89.2857%	Val Loss: 1.747868	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.384889	Train Acc: 91.4286%	Val Loss: 1.756737	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.369540	Train Acc: 91.4286%	Val Loss: 1.761198	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  30	Train Loss: 0.354952	Train Acc: 91.4286%	Val Loss: 1.763181	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  31	Train Loss: 0.341409	Train Acc: 91.4286%	Val Loss: 1.763475	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  32	Train Loss: 0.328835	Train Acc: 91.4286%	Val Loss: 1.765086	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  33	Train Loss: 0.316971	Train Acc: 92.1429%	Val Loss: 1.768006	Val Acc: 65.0000%</span></span>
<span class="line"><span>Early Stopping at Epoch 33</span></span>
<span class="line"><span>2025-04-07 02:16:15.243482: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:16:15.506881: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:16:15.754760: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:16:15.954381: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743992176.300365 3890942 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743992176.300408 3890942 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743992176.300416 3890942 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743992176.300424 3890942 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.300432 3890942 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743992176.300440 3890942 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743992176.300447 3890942 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.300455 3890942 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743992176.300462 3890942 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.300470 3890942 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-07 02:16:16.300482: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.303064 3890942 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743992176.303091 3890942 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743992176.303099 3890942 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743992176.303107 3890942 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.303114 3890942 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743992176.303122 3890942 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743992176.303129 3890942 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.303137 3890942 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743992176.303144 3890942 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.303151 3890942 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-07 02:16:16.303163: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.305751 3890942 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743992176.305778 3890942 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743992176.305786 3890942 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743992176.305793 3890942 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.305801 3890942 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743992176.305808 3890942 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743992176.305820 3890942 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.305827 3890942 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743992176.305834 3890942 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.305842 3890942 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-07 02:16:16.305853: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.308422 3890942 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743992176.308434 3890942 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743992176.308437 3890942 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743992176.308440 3890942 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.308443 3890942 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743992176.308446 3890942 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743992176.308449 3890942 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.308452 3890942 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743992176.308455 3890942 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.308458 3890942 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-07 02:16:16.308463: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.310729 3890942 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743992176.310740 3890942 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743992176.310744 3890942 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743992176.310747 3890942 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.310750 3890942 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743992176.310753 3890942 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743992176.310756 3890942 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.310759 3890942 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743992176.310762 3890942 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.310765 3890942 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-07 02:16:16.310770: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.313057 3890942 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743992176.313068 3890942 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743992176.313071 3890942 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743992176.313074 3890942 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.313077 3890942 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743992176.313080 3890942 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743992176.313083 3890942 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.313086 3890942 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743992176.313091 3890942 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.313094 3890942 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-04-07 02:16:16.313099: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.315338 3890942 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1743992176.315349 3890942 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743992176.315352 3890942 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743992176.315355 3890942 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743992176.315358 3890942 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743992176.315361 3890942 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743992176.315364 3890942 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743992176.315367 3890942 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1743992176.315370 3890942 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743992176.315373 3890942 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-04-07 02:16:16.315378: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.317601 3890942 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743992176.317612 3890942 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743992176.317615 3890942 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743992176.317618 3890942 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743992176.317621 3890942 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743992176.317624 3890942 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743992176.317628 3890942 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743992176.317630 3890942 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1743992176.317633 3890942 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1743992176.317636 3890942 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-04-07 02:16:16.317641: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.319906 3890942 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743992176.319917 3890942 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743992176.319920 3890942 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743992176.319923 3890942 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743992176.319926 3890942 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743992176.319929 3890942 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743992176.319932 3890942 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743992176.319935 3890942 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.319938 3890942 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743992176.319941 3890942 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-07 02:16:16.319948: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.322161 3890942 buffer_comparator.cc:156] Difference at 7: 1059.63, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1743992176.322172 3890942 buffer_comparator.cc:156] Difference at 11: 1264.78, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1743992176.322177 3890942 buffer_comparator.cc:156] Difference at 179: 1224.64, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1743992176.322180 3890942 buffer_comparator.cc:156] Difference at 266: 1048.08, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1743992176.322183 3890942 buffer_comparator.cc:156] Difference at 270: 1247.68, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1743992176.322187 3890942 buffer_comparator.cc:156] Difference at 417: 1223.32, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1743992176.322190 3890942 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743992176.322193 3890942 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743992176.322197 3890942 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743992176.322199 3890942 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-04-07 02:16:16.322204: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.324462 3890942 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743992176.324473 3890942 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743992176.324476 3890942 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743992176.324479 3890942 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743992176.324482 3890942 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743992176.324485 3890942 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743992176.324488 3890942 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743992176.324491 3890942 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.324494 3890942 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743992176.324497 3890942 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-07 02:16:16.324502: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.326770 3890942 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743992176.326783 3890942 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743992176.326786 3890942 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743992176.326789 3890942 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743992176.326792 3890942 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743992176.326795 3890942 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743992176.326798 3890942 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743992176.326801 3890942 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743992176.326804 3890942 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743992176.326807 3890942 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-04-07 02:16:16.326812: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.329091 3890942 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992176.329103 3890942 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743992176.329106 3890942 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743992176.329109 3890942 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992176.329113 3890942 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743992176.329116 3890942 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.329119 3890942 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.329122 3890942 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743992176.329125 3890942 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.329128 3890942 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-07 02:16:16.329132: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.331388 3890942 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992176.331401 3890942 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743992176.331404 3890942 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743992176.331407 3890942 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992176.331410 3890942 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743992176.331413 3890942 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.331416 3890942 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.331419 3890942 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743992176.331422 3890942 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.331425 3890942 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-07 02:16:16.331430: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.333692 3890942 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992176.333703 3890942 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743992176.333706 3890942 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743992176.333709 3890942 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992176.333712 3890942 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743992176.333715 3890942 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.333718 3890942 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.333721 3890942 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743992176.333724 3890942 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.333727 3890942 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-07 02:16:16.333732: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.335973 3890942 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1743992176.335987 3890942 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992176.335990 3890942 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743992176.335993 3890942 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743992176.335996 3890942 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992176.335999 3890942 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743992176.336002 3890942 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.336005 3890942 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1743992176.336008 3890942 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.336011 3890942 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-04-07 02:16:16.336016: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.338287 3890942 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743992176.338301 3890942 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743992176.338304 3890942 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743992176.338307 3890942 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743992176.338310 3890942 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743992176.338313 3890942 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743992176.338316 3890942 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.338319 3890942 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743992176.338322 3890942 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743992176.338325 3890942 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-04-07 02:16:16.338330: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.340686 3890942 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743992176.340699 3890942 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743992176.340703 3890942 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743992176.340706 3890942 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.340709 3890942 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743992176.340712 3890942 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743992176.340715 3890942 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743992176.340718 3890942 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1743992176.340721 3890942 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1743992176.340724 3890942 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-04-07 02:16:16.340728: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992176.343107 3890942 buffer_comparator.cc:156] Difference at 1793: 1450.89, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743992176.343120 3890942 buffer_comparator.cc:156] Difference at 1794: 1267.6, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743992176.343123 3890942 buffer_comparator.cc:156] Difference at 1795: 881.963, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743992176.343128 3890942 buffer_comparator.cc:156] Difference at 1796: 1413.49, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743992176.343131 3890942 buffer_comparator.cc:156] Difference at 1797: 1005.6, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743992176.343134 3890942 buffer_comparator.cc:156] Difference at 1798: 764.123, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743992176.343137 3890942 buffer_comparator.cc:156] Difference at 1800: 1466.23, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743992176.343140 3890942 buffer_comparator.cc:156] Difference at 1801: 1286.98, expected 1808.37</span></span>
<span class="line"><span>E0000 00:00:1743992176.343143 3890942 buffer_comparator.cc:156] Difference at 1802: 899.199, expected 1570.73</span></span>
<span class="line"><span>E0000 00:00:1743992176.343146 3890942 buffer_comparator.cc:156] Difference at 1803: 1441.04, expected 1102.47</span></span>
<span class="line"><span>2025-04-07 02:16:16.343151: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-04-07 02:16:17.215801: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:16:17.934318: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:16:18.017298: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 172 bytes spill stores, 172 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-04-07 02:16:18.078839: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 8 bytes spill stores, 8 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743992178.214226 3890942 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743992178.214295 3890942 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743992178.214303 3890942 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743992178.214311 3890942 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.214318 3890942 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743992178.214326 3890942 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743992178.214333 3890942 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743992178.214340 3890942 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743992178.214347 3890942 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.214354 3890942 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-07 02:16:18.214370: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.216750 3890942 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743992178.216767 3890942 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743992178.216772 3890942 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743992178.216776 3890942 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.216780 3890942 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743992178.216784 3890942 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743992178.216788 3890942 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743992178.216792 3890942 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743992178.216796 3890942 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.216803 3890942 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-07 02:16:18.216809: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.219178 3890942 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743992178.219196 3890942 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743992178.219200 3890942 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743992178.219204 3890942 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.219209 3890942 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743992178.219213 3890942 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743992178.219217 3890942 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743992178.219221 3890942 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743992178.219225 3890942 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.219229 3890942 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-07 02:16:18.219235: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.221710 3890942 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743992178.221726 3890942 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743992178.221731 3890942 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743992178.221735 3890942 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.221739 3890942 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743992178.221743 3890942 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743992178.221747 3890942 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743992178.221751 3890942 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743992178.221755 3890942 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.221759 3890942 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-07 02:16:18.221766: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.224154 3890942 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743992178.224172 3890942 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743992178.224176 3890942 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743992178.224180 3890942 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.224184 3890942 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743992178.224189 3890942 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743992178.224193 3890942 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743992178.224197 3890942 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743992178.224201 3890942 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.224205 3890942 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-07 02:16:18.224211: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.226617 3890942 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743992178.226637 3890942 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743992178.226642 3890942 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743992178.226646 3890942 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.226650 3890942 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743992178.226654 3890942 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743992178.226658 3890942 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743992178.226662 3890942 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743992178.226666 3890942 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.226670 3890942 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-04-07 02:16:18.226677: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.228946 3890942 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743992178.228958 3890942 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743992178.228961 3890942 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743992178.228964 3890942 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743992178.228967 3890942 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743992178.228970 3890942 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743992178.228973 3890942 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743992178.228976 3890942 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743992178.228979 3890942 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743992178.228982 3890942 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-07 02:16:18.228986: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.231224 3890942 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743992178.231237 3890942 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743992178.231240 3890942 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743992178.231243 3890942 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743992178.231246 3890942 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743992178.231249 3890942 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743992178.231251 3890942 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743992178.231254 3890942 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743992178.231257 3890942 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743992178.231260 3890942 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-04-07 02:16:18.231265: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.233535 3890942 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743992178.233547 3890942 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743992178.233550 3890942 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743992178.233553 3890942 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743992178.233556 3890942 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743992178.233561 3890942 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743992178.233564 3890942 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743992178.233567 3890942 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743992178.233570 3890942 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743992178.233572 3890942 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-07 02:16:18.233577: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.235814 3890942 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743992178.235827 3890942 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743992178.235830 3890942 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743992178.235833 3890942 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743992178.235836 3890942 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743992178.235839 3890942 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743992178.235842 3890942 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743992178.235845 3890942 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743992178.235849 3890942 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743992178.235853 3890942 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-07 02:16:18.235857: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.238325 3890942 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743992178.238338 3890942 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743992178.238341 3890942 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743992178.238344 3890942 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743992178.238347 3890942 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743992178.238350 3890942 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743992178.238353 3890942 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743992178.238356 3890942 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743992178.238359 3890942 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743992178.238361 3890942 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-07 02:16:18.238366: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.240658 3890942 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743992178.240671 3890942 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743992178.240674 3890942 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743992178.240677 3890942 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743992178.240680 3890942 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743992178.240683 3890942 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743992178.240686 3890942 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743992178.240689 3890942 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743992178.240692 3890942 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743992178.240696 3890942 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-04-07 02:16:18.240701: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.242969 3890942 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743992178.242982 3890942 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.242985 3890942 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.242988 3890942 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743992178.242991 3890942 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743992178.242994 3890942 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743992178.242997 3890942 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743992178.243000 3890942 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743992178.243002 3890942 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743992178.243005 3890942 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-07 02:16:18.243010: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.245272 3890942 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743992178.245284 3890942 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.245288 3890942 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.245291 3890942 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743992178.245294 3890942 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743992178.245296 3890942 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743992178.245299 3890942 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743992178.245302 3890942 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743992178.245305 3890942 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743992178.245308 3890942 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-07 02:16:18.245312: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.247589 3890942 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743992178.247602 3890942 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.247605 3890942 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.247608 3890942 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743992178.247611 3890942 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743992178.247614 3890942 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743992178.247617 3890942 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743992178.247620 3890942 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743992178.247623 3890942 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743992178.247625 3890942 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-07 02:16:18.247630: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.249881 3890942 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743992178.249895 3890942 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.249898 3890942 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.249901 3890942 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743992178.249904 3890942 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743992178.249907 3890942 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743992178.249910 3890942 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743992178.249913 3890942 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743992178.249916 3890942 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743992178.249918 3890942 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-07 02:16:18.249923: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.252168 3890942 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743992178.252181 3890942 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743992178.252184 3890942 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743992178.252187 3890942 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743992178.252190 3890942 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743992178.252192 3890942 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743992178.252195 3890942 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743992178.252198 3890942 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743992178.252201 3890942 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743992178.252204 3890942 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-04-07 02:16:18.252208: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.254585 3890942 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743992178.254597 3890942 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743992178.254600 3890942 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743992178.254603 3890942 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743992178.254606 3890942 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743992178.254609 3890942 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743992178.254612 3890942 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743992178.254615 3890942 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743992178.254618 3890942 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743992178.254621 3890942 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-04-07 02:16:18.254625: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743992178.256990 3890942 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743992178.257002 3890942 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743992178.257005 3890942 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743992178.257008 3890942 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743992178.257011 3890942 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743992178.257015 3890942 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743992178.257018 3890942 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743992178.257021 3890942 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743992178.257024 3890942 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743992178.257027 3890942 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-04-07 02:16:18.257032: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.649787	Test Acc: 67.6000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
